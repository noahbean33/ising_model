use rust_core::add;

#[test]
fn test_add_from_integration() {
    // This test calls the public `add` function from the `rust_core` library.
    assert_eq!(add(2, 2), 4);
}
