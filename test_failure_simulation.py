import json
import os
import urllib.request


URL = os.environ.get("TEST_URL", "http://localhost:5000/ask")


def main():
    payload = {
        "id": 1,
        "query": "What is middleware in distributed systems?"
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("Sending one request. The first worker should timeout, then the task should be reassigned.")

    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
        print("Response status:", response.status)
        print("Response body:")
        print(body)


if __name__ == "__main__":
    main()
