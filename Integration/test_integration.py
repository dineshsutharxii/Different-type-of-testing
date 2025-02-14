from business_logic import calculate_year_of_birth, find_name


def test_integration_for_dob():
    result = calculate_year_of_birth(1)
    assert result == 1994, f"Expected 1994, but got {result}"
    result = calculate_year_of_birth(2)
    assert result == 1999, f"Expected 1999, but got {result}"
    result = calculate_year_of_birth(999)
    assert result is None, f"Expected None, but got {result}"


def test_integration_name():
    result = find_name(1)
    assert result == 'Alice', f"Expected Alice, but got {result}"
    result = find_name(2)
    assert result == 'Bob', f"Expected Bob, but got {result}"
    result = find_name(999)
    assert result is None, f"Expected None, but got {result}"
