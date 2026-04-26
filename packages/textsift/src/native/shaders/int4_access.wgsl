fn load_byte(arr: ptr<storage, array<u32>, read>, byte_idx: u32) -> u32 {
    let word = (*arr)[byte_idx >> 2u];
    let shift = (byte_idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn load_nibble(arr: ptr<storage, array<u32>, read>, nibble_idx: u32) -> u32 {
    let byte_val = load_byte(arr, nibble_idx >> 1u);
    let hi = (nibble_idx & 1u) == 1u;
    return select(byte_val & 0xFu, (byte_val >> 4u) & 0xFu, hi);
}
