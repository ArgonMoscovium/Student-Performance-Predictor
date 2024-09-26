from app import app

def test_home():
    # Create a test client using Flask's built-in test client
    client = app.test_client()

    # Use the test client to make a GET request to the root endpoint ("/")
    response = client.get("/")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Check if specific content or elements from 'index.html' are in the response data
    # Since you're rendering HTML templates, checking for a part of the HTML content is appropriate
    assert b"<h1>This is the home page.<h1>" in response.data  # Change this to match your index.html content

def test_predict_get():
    client = app.test_client()

    # Send a GET request to the '/predictdata' route
    response = client.get("/predictdata")

    # Check if the response status is OK (200)
    assert response.status_code == 200

    # Optionally, check for specific HTML elements in the home.html page
    assert b"<form" in response.data  # Verifying if a form exists in 'home.html'

def test_predict_post():
    client = app.test_client()

    # Simulate form data submission via POST request to '/predictdata'
    form_data = {
        "gender": "male",
        "race_ethnicity": "group A",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": 78,
        "writing_score": 82
    }

    response = client.post("/predictdata", data=form_data)

    # Check if the POST request returns a successful status
    assert response.status_code == 200

    # Check if the result appears in the response 
    assert b"THE prediction is" in response.data  # Update based on your actual output

