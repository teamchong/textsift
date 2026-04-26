// Vulkan-direct bridge implementation. Mirrors metal/bridge.m's API
// shape so the Zig wrapper layer is platform-agnostic.
//
// Lifecycle:
//   ts_vk_create_backend() — VkInstance + physical device + queue family
//                            + VkDevice + queue. Single global backend.
//   ts_vk_pipeline_create() — VkShaderModule + per-pipeline DSL +
//                             VkPipelineLayout + VkComputePipeline.
//                             Cached by name in the Zig layer.
//   ts_vk_buffer_create() — HOST_VISIBLE | HOST_COHERENT VkBuffer +
//                           VkDeviceMemory, persistently mapped.
//   ts_vk_encoder_begin() — VkCommandPool + VkCommandBuffer +
//                           VkDescriptorPool + VkFence. begin recording.
//   ts_vk_encoder_dispatch() — alloc DescriptorSet, write buffer bindings,
//                              bind pipeline + set, push constants,
//                              dispatch, then emit a compute→compute
//                              write→read pipeline barrier.
//   ts_vk_encoder_submit_and_wait() — end recording, submit, fence-wait.
//
// Validation layers: enabled if env TEXTSIFT_VK_VALIDATION=1. For dev/CI;
// off in shipped builds (perf overhead).

#include "bridge.h"

#include <vulkan/vulkan.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// ── small utilities ────────────────────────────────────────────────────

static void format_err(char* err_out, size_t err_cap, const char* fmt, ...) {
    if (!err_out || err_cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err_out, err_cap, fmt, ap);
    va_end(ap);
}

static const char* vk_result_str(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        default: return "VK_ERROR_UNKNOWN";
    }
}

#define VK_CHECK_OR_RETURN(expr, ret, err, cap, where) do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
        format_err((err), (cap), "%s: %s (%d)", (where), vk_result_str(_r), _r); \
        return (ret); \
    } \
} while (0)

// ── opaque struct definitions ──────────────────────────────────────────

struct TsVkBackendImpl {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t queue_family;
    VkDevice device;
    VkQueue queue;

    // Cached for fast memory-type lookup. Iris Xe / Mesa exposes a
    // memory type that's DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
    // (UMA). Discrete GPUs typically only have HOST_VISIBLE+HOST_COHERENT
    // on a system-RAM-backed type. We pick HOST_VISIBLE+HOST_COHERENT
    // either way (UMA gets device-local for free).
    int32_t mem_type_host_visible_coherent;

    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
};

struct TsVkBufferImpl {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t length;
    void* mapped;
};

struct TsVkPipelineImpl {
    VkShaderModule shader;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    uint32_t num_storage_buffers;
    uint32_t push_constant_size;
};

struct TsVkEncoderImpl {
    TsVkBackend backend;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buffer;
    VkDescriptorPool desc_pool;
    VkFence fence;
    bool recording;
};

// ── backend lifecycle ──────────────────────────────────────────────────

static int32_t find_memory_type(
    const VkPhysicalDeviceMemoryProperties* mp,
    uint32_t type_bits,
    VkMemoryPropertyFlags req)
{
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) == 0) continue;
        if ((mp->memoryTypes[i].propertyFlags & req) == req) return (int32_t)i;
    }
    return -1;
}

static int validation_enabled(void) {
    const char* v = getenv("TEXTSIFT_VK_VALIDATION");
    return v && v[0] == '1';
}

TsVkBackend ts_vk_create_backend(char* err_out, size_t err_cap) {
    TsVkBackend b = (TsVkBackend)calloc(1, sizeof(*b));
    if (!b) { format_err(err_out, err_cap, "calloc backend failed"); return NULL; }

    // ── instance ──
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "textsift",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "textsift-vulkan-direct",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    uint32_t num_layers = validation_enabled() ? 1 : 0;

    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledLayerCount = num_layers,
        .ppEnabledLayerNames = num_layers ? layers : NULL,
    };
    VK_CHECK_OR_RETURN(vkCreateInstance(&ici, NULL, &b->instance),
                       (free(b), NULL), err_out, err_cap, "vkCreateInstance");

    // ── physical device ──
    uint32_t pd_count = 0;
    vkEnumeratePhysicalDevices(b->instance, &pd_count, NULL);
    if (pd_count == 0) {
        format_err(err_out, err_cap, "no Vulkan physical devices");
        vkDestroyInstance(b->instance, NULL); free(b); return NULL;
    }
    VkPhysicalDevice* pds = (VkPhysicalDevice*)calloc(pd_count, sizeof(*pds));
    vkEnumeratePhysicalDevices(b->instance, &pd_count, pds);

    // Score: discrete > integrated > virtual > cpu > other. Pick highest.
    int best_score = -1;
    int best_idx = -1;
    for (uint32_t i = 0; i < pd_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pds[i], &props);
        int score = 0;
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 4; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 3; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score = 2; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: score = 1; break;
            default: score = 0;
        }
        if (score > best_score) { best_score = score; best_idx = (int)i; }
    }
    if (best_idx < 0) {
        format_err(err_out, err_cap, "no suitable physical device");
        free(pds); vkDestroyInstance(b->instance, NULL); free(b); return NULL;
    }
    b->physical_device = pds[best_idx];
    VkPhysicalDeviceProperties dprops;
    vkGetPhysicalDeviceProperties(b->physical_device, &dprops);
    snprintf(b->device_name, sizeof(b->device_name), "%s", dprops.deviceName);
    free(pds);

    // ── queue family (compute) ──
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(b->physical_device, &qf_count, NULL);
    VkQueueFamilyProperties* qfs = (VkQueueFamilyProperties*)calloc(qf_count, sizeof(*qfs));
    vkGetPhysicalDeviceQueueFamilyProperties(b->physical_device, &qf_count, qfs);
    uint32_t qf_idx = UINT32_MAX;
    // Prefer a dedicated compute queue (compute set, graphics not set);
    // fall back to any queue with compute support.
    for (uint32_t i = 0; i < qf_count; i++) {
        if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qf_idx = i; break; }
    }
    if (qf_idx == UINT32_MAX) {
        for (uint32_t i = 0; i < qf_count; i++) {
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf_idx = i; break; }
        }
    }
    free(qfs);
    if (qf_idx == UINT32_MAX) {
        format_err(err_out, err_cap, "no compute queue family on '%s'", b->device_name);
        vkDestroyInstance(b->instance, NULL); free(b); return NULL;
    }
    b->queue_family = qf_idx;

    // ── logical device + queue, with f16 + 16-bit storage features ──
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf_idx,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };

    // Vulkan 1.2 chains: shaderFloat16 from VkPhysicalDeviceVulkan12Features,
    // storageBuffer16BitAccess from VkPhysicalDeviceVulkan11Features.
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .storageBuffer16BitAccess = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f11,
        .shaderFloat16 = VK_TRUE,
    };

    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f12,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VK_CHECK_OR_RETURN(vkCreateDevice(b->physical_device, &dci, NULL, &b->device),
                       (vkDestroyInstance(b->instance, NULL), free(b), NULL),
                       err_out, err_cap, "vkCreateDevice");
    vkGetDeviceQueue(b->device, qf_idx, 0, &b->queue);

    // ── memory props (cached for buffer allocation) ──
    vkGetPhysicalDeviceMemoryProperties(b->physical_device, &b->mem_props);
    b->mem_type_host_visible_coherent = find_memory_type(
        &b->mem_props,
        UINT32_MAX,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (b->mem_type_host_visible_coherent < 0) {
        format_err(err_out, err_cap, "no HOST_VISIBLE | HOST_COHERENT memory type on '%s'", b->device_name);
        vkDestroyDevice(b->device, NULL); vkDestroyInstance(b->instance, NULL); free(b); return NULL;
    }

    return b;
}

void ts_vk_destroy_backend(TsVkBackend b) {
    if (!b) return;
    if (b->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(b->device);
        vkDestroyDevice(b->device, NULL);
    }
    if (b->instance != VK_NULL_HANDLE) vkDestroyInstance(b->instance, NULL);
    free(b);
}

const char* ts_vk_device_name(TsVkBackend b) {
    return b ? b->device_name : "";
}

// ── pipelines ──────────────────────────────────────────────────────────

TsVkPipeline ts_vk_pipeline_create(
    TsVkBackend b,
    const uint32_t* spv, size_t spv_len_bytes,
    const char* entry_point,
    uint32_t num_storage_buffers,
    uint32_t push_constant_size,
    char* err_out, size_t err_cap)
{
    if (!b || !spv || spv_len_bytes == 0) {
        format_err(err_out, err_cap, "ts_vk_pipeline_create: null arg");
        return NULL;
    }

    TsVkPipeline p = (TsVkPipeline)calloc(1, sizeof(*p));
    if (!p) { format_err(err_out, err_cap, "calloc pipeline failed"); return NULL; }
    p->num_storage_buffers = num_storage_buffers;
    p->push_constant_size = push_constant_size;

    // ── shader module ──
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_len_bytes,
        .pCode = spv,
    };
    VK_CHECK_OR_RETURN(vkCreateShaderModule(b->device, &smci, NULL, &p->shader),
                       (free(p), NULL), err_out, err_cap, "vkCreateShaderModule");

    // ── descriptor set layout: one set, N storage buffer bindings ──
    VkDescriptorSetLayoutBinding bindings[16];
    if (num_storage_buffers > 16) {
        format_err(err_out, err_cap, "pipeline: too many storage buffers (%u > 16)", num_storage_buffers);
        vkDestroyShaderModule(b->device, p->shader, NULL); free(p); return NULL;
    }
    for (uint32_t i = 0; i < num_storage_buffers; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = num_storage_buffers,
        .pBindings = num_storage_buffers ? bindings : NULL,
    };
    VK_CHECK_OR_RETURN(vkCreateDescriptorSetLayout(b->device, &dslci, NULL, &p->dsl),
                       (vkDestroyShaderModule(b->device, p->shader, NULL), free(p), NULL),
                       err_out, err_cap, "vkCreateDescriptorSetLayout");

    // ── pipeline layout (descriptor sets + push constants) ──
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &p->dsl,
        .pushConstantRangeCount = push_constant_size > 0 ? 1 : 0,
        .pPushConstantRanges = push_constant_size > 0 ? &pcr : NULL,
    };
    VK_CHECK_OR_RETURN(vkCreatePipelineLayout(b->device, &plci, NULL, &p->layout),
                       (vkDestroyDescriptorSetLayout(b->device, p->dsl, NULL),
                        vkDestroyShaderModule(b->device, p->shader, NULL),
                        free(p), NULL),
                       err_out, err_cap, "vkCreatePipelineLayout");

    // ── compute pipeline ──
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = p->shader,
            .pName = entry_point ? entry_point : "main",
        },
        .layout = p->layout,
    };
    VK_CHECK_OR_RETURN(vkCreateComputePipelines(b->device, VK_NULL_HANDLE, 1, &cpci, NULL, &p->pipeline),
                       (vkDestroyPipelineLayout(b->device, p->layout, NULL),
                        vkDestroyDescriptorSetLayout(b->device, p->dsl, NULL),
                        vkDestroyShaderModule(b->device, p->shader, NULL),
                        free(p), NULL),
                       err_out, err_cap, "vkCreateComputePipelines");

    return p;
}

void ts_vk_pipeline_release(TsVkBackend b, TsVkPipeline p) {
    if (!p || !b) return;
    if (p->pipeline != VK_NULL_HANDLE) vkDestroyPipeline(b->device, p->pipeline, NULL);
    if (p->layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(b->device, p->layout, NULL);
    if (p->dsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(b->device, p->dsl, NULL);
    if (p->shader != VK_NULL_HANDLE) vkDestroyShaderModule(b->device, p->shader, NULL);
    free(p);
}

// ── buffers ────────────────────────────────────────────────────────────

TsVkBuffer ts_vk_buffer_create(TsVkBackend b, size_t length, const void* init_data) {
    if (!b || length == 0) return NULL;
    TsVkBuffer buf = (TsVkBuffer)calloc(1, sizeof(*buf));
    if (!buf) return NULL;
    buf->length = length;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = length,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
               | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
               | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(b->device, &bci, NULL, &buf->buffer) != VK_SUCCESS) {
        free(buf); return NULL;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(b->device, buf->buffer, &req);

    int32_t mt = find_memory_type(
        &b->mem_props, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt < 0) {
        vkDestroyBuffer(b->device, buf->buffer, NULL);
        free(buf); return NULL;
    }

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = (uint32_t)mt,
    };
    if (vkAllocateMemory(b->device, &mai, NULL, &buf->memory) != VK_SUCCESS) {
        vkDestroyBuffer(b->device, buf->buffer, NULL);
        free(buf); return NULL;
    }
    if (vkBindBufferMemory(b->device, buf->buffer, buf->memory, 0) != VK_SUCCESS) {
        vkFreeMemory(b->device, buf->memory, NULL);
        vkDestroyBuffer(b->device, buf->buffer, NULL);
        free(buf); return NULL;
    }
    if (vkMapMemory(b->device, buf->memory, 0, VK_WHOLE_SIZE, 0, &buf->mapped) != VK_SUCCESS) {
        vkFreeMemory(b->device, buf->memory, NULL);
        vkDestroyBuffer(b->device, buf->buffer, NULL);
        free(buf); return NULL;
    }
    if (init_data) {
        memcpy(buf->mapped, init_data, length);
    } else {
        memset(buf->mapped, 0, length);
    }
    return buf;
}

void ts_vk_buffer_release(TsVkBackend b, TsVkBuffer buf) {
    if (!buf || !b) return;
    if (buf->mapped) vkUnmapMemory(b->device, buf->memory);
    if (buf->memory != VK_NULL_HANDLE) vkFreeMemory(b->device, buf->memory, NULL);
    if (buf->buffer != VK_NULL_HANDLE) vkDestroyBuffer(b->device, buf->buffer, NULL);
    free(buf);
}

void ts_vk_buffer_write(TsVkBuffer buf, size_t offset, const void* data, size_t len) {
    if (!buf || !buf->mapped || !data) return;
    if (offset + len > buf->length) return;
    memcpy((char*)buf->mapped + offset, data, len);
}

void ts_vk_buffer_read(TsVkBuffer buf, size_t offset, void* dst, size_t len) {
    if (!buf || !buf->mapped || !dst) return;
    if (offset + len > buf->length) return;
    memcpy(dst, (char*)buf->mapped + offset, len);
}

size_t ts_vk_buffer_length(TsVkBuffer buf) {
    return buf ? buf->length : 0;
}

// ── encoder ────────────────────────────────────────────────────────────

// Pool sizing: each encoder hosts one forward pass (~133 dispatches).
// Per dispatch: 1 descriptor set, ≤16 storage buffer descriptors.
// Headroom: 1024 sets * 16 binds = 16K descriptors.
#define DESC_POOL_SETS 1024
#define DESC_POOL_BUFFERS 16384

TsVkEncoder ts_vk_encoder_begin(TsVkBackend b) {
    if (!b) return NULL;
    TsVkEncoder e = (TsVkEncoder)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->backend = b;

    // ── command pool + buffer ──
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = b->queue_family,
    };
    if (vkCreateCommandPool(b->device, &cpci, NULL, &e->cmd_pool) != VK_SUCCESS) {
        free(e); return NULL;
    }

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = e->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(b->device, &cbai, &e->cmd_buffer) != VK_SUCCESS) {
        vkDestroyCommandPool(b->device, e->cmd_pool, NULL);
        free(e); return NULL;
    }

    // ── descriptor pool ──
    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = DESC_POOL_BUFFERS,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = DESC_POOL_SETS,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    if (vkCreateDescriptorPool(b->device, &dpci, NULL, &e->desc_pool) != VK_SUCCESS) {
        vkDestroyCommandPool(b->device, e->cmd_pool, NULL);
        free(e); return NULL;
    }

    // ── fence ──
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(b->device, &fci, NULL, &e->fence) != VK_SUCCESS) {
        vkDestroyDescriptorPool(b->device, e->desc_pool, NULL);
        vkDestroyCommandPool(b->device, e->cmd_pool, NULL);
        free(e); return NULL;
    }

    // ── begin recording ──
    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(e->cmd_buffer, &cbbi) != VK_SUCCESS) {
        vkDestroyFence(b->device, e->fence, NULL);
        vkDestroyDescriptorPool(b->device, e->desc_pool, NULL);
        vkDestroyCommandPool(b->device, e->cmd_pool, NULL);
        free(e); return NULL;
    }
    e->recording = true;
    return e;
}

void ts_vk_encoder_release(TsVkEncoder e) {
    if (!e) return;
    if (e->recording) vkEndCommandBuffer(e->cmd_buffer);
    if (e->fence != VK_NULL_HANDLE) vkDestroyFence(e->backend->device, e->fence, NULL);
    if (e->desc_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(e->backend->device, e->desc_pool, NULL);
    if (e->cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(e->backend->device, e->cmd_pool, NULL);
    free(e);
}

void ts_vk_encoder_dispatch(
    TsVkEncoder e,
    TsVkPipeline p,
    const TsVkBuffer* bindings, uint32_t num_bindings,
    const void* push_data, uint32_t push_data_len,
    uint32_t gx, uint32_t gy, uint32_t gz)
{
    if (!e || !p || !e->recording) return;
    if (num_bindings != p->num_storage_buffers) {
        fprintf(stderr, "ts_vk_encoder_dispatch: binding count mismatch (got %u, pipeline expects %u)\n",
                num_bindings, p->num_storage_buffers);
        return;
    }
    if (push_data_len != p->push_constant_size) {
        fprintf(stderr, "ts_vk_encoder_dispatch: push-constant size mismatch (got %u, pipeline expects %u)\n",
                push_data_len, p->push_constant_size);
        return;
    }

    VkDevice dev = e->backend->device;

    // Allocate a fresh descriptor set per dispatch.
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = e->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &p->dsl,
    };
    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(dev, &dsai, &ds) != VK_SUCCESS) {
        fprintf(stderr, "ts_vk_encoder_dispatch: descriptor pool exhausted\n");
        return;
    }

    // Update bindings.
    VkDescriptorBufferInfo dbis[16];
    VkWriteDescriptorSet writes[16];
    for (uint32_t i = 0; i < num_bindings; i++) {
        if (!bindings[i]) {
            fprintf(stderr, "ts_vk_encoder_dispatch: null binding at slot %u\n", i);
            return;
        }
        dbis[i] = (VkDescriptorBufferInfo){
            .buffer = bindings[i]->buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds,
            .dstBinding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &dbis[i],
        };
    }
    vkUpdateDescriptorSets(dev, num_bindings, writes, 0, NULL);

    // Bind pipeline + descriptor set + push constants, then dispatch.
    vkCmdBindPipeline(e->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(e->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p->layout, 0, 1, &ds, 0, NULL);
    if (p->push_constant_size > 0 && push_data) {
        vkCmdPushConstants(e->cmd_buffer, p->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, p->push_constant_size, push_data);
    }
    vkCmdDispatch(e->cmd_buffer, gx, gy, gz);

    // Compute→compute write→read barrier so the next dispatch sees this
    // dispatch's writes. Conservative (covers all SSBO writes); cheap.
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(e->cmd_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1, &mb, 0, NULL, 0, NULL);
}

void ts_vk_encoder_submit_and_wait(TsVkEncoder e) {
    if (!e) return;
    VkDevice dev = e->backend->device;

    if (e->recording) {
        vkEndCommandBuffer(e->cmd_buffer);
        e->recording = false;
    }

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &e->cmd_buffer,
    };
    if (vkQueueSubmit(e->backend->queue, 1, &si, e->fence) != VK_SUCCESS) {
        fprintf(stderr, "ts_vk_encoder_submit_and_wait: vkQueueSubmit failed\n");
        ts_vk_encoder_release(e);
        return;
    }

    // Wait up to 30s. UMA + integrated forward should be milliseconds;
    // 30s is a safety net for first-time pipeline cold-start under
    // validation layers.
    vkWaitForFences(dev, 1, &e->fence, VK_TRUE, 30ULL * 1000ULL * 1000ULL * 1000ULL);

    vkDestroyFence(dev, e->fence, NULL); e->fence = VK_NULL_HANDLE;
    vkDestroyDescriptorPool(dev, e->desc_pool, NULL); e->desc_pool = VK_NULL_HANDLE;
    vkDestroyCommandPool(dev, e->cmd_pool, NULL); e->cmd_pool = VK_NULL_HANDLE;
    free(e);
}
