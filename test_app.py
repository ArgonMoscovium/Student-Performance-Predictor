from app import app
def test_home():
    response = app.test_client.get("/")

    assert response.status_code == 200 # assert keyword when debugging code, , to test if a condition in the code returns True
    assert response.data == b"Hello Anurag!"