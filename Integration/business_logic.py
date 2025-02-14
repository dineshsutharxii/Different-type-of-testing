from db_module import fetch_user_data


def calculate_year_of_birth(user_id):
    """Calculate the year of birth based on user data."""
    user_data = fetch_user_data(user_id)
    if user_data:
        current_year = 2024
        return current_year - user_data['age']
    else:
        return None


def find_name(user_id):
    """Calculate the year of birth based on user data."""
    user_data = fetch_user_data(user_id)
    if user_data:
        return user_data['name']
    else:
        return None
