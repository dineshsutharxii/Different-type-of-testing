def fetch_user_data(user_id):
    """Fetch user data from a mock database."""
    # Simulating a database with a dictionary
    database = {
        1: {'name': 'Alice', 'age': 30},
        2: {'name': 'Bob', 'age': 25},
        3: {'name': 'Charlie', 'age': 35}
    }
    return database.get(user_id)
