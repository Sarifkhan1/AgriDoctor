import requests
import sys
import time
import uuid

BASE_URL = "http://localhost:8000/api"

def test_auth():
    print("🧪 Testing Authentication...")
    
    # Generate unique user
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword123"
    
    # 1. Register
    print(f"\n1. Registering user: {email}")
    try:
        resp = requests.post(f"{BASE_URL}/auth/register", json={
            "email": email,
            "password": password
        })
        if resp.status_code != 200:
            print(f"❌ Registration failed: {resp.status_code} - {resp.text}")
            return False
        data = resp.json()
        token = data.get("access_token")
        if not token:
            print("❌ No access token in register response")
            return False
        print("✅ Registration successful")
    except Exception as e:
        print(f"❌ Registration exception: {e}")
        return False

    # 2. Login
    print(f"\n2. Logging in...")
    try:
        resp = requests.post(f"{BASE_URL}/auth/login", json={
            "email": email,
            "password": password
        })
        if resp.status_code != 200:
            print(f"❌ Login failed: {resp.status_code} - {resp.text}")
            return False
        data = resp.json()
        login_token = data.get("access_token")
        if not login_token:
            print("❌ No access token in login response")
            return False
        print("✅ Login successful")
    except Exception as e:
        print(f"❌ Login exception: {e}")
        return False

    # 3. Protected Route
    print(f"\n3. Testing protected route (/auth/me)...")
    try:
        headers = {"Authorization": f"Bearer {login_token}"}
        resp = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        if resp.status_code != 200:
            print(f"❌ Protected route failed: {resp.status_code} - {resp.text}")
            return False
        user_data = resp.json()
        if user_data.get("email") != email:
            print(f"❌ User data mismatch: expected {email}, got {user_data.get('email')}")
            return False
        print("✅ Protected route successful")
    except Exception as e:
        print(f"❌ Protected route exception: {e}")
        return False

    print("\n🎉 All Auth Tests Passed!")
    return True

if __name__ == "__main__":
    try:
        if test_auth():
            sys.exit(0)
        else:
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Is it running?")
        sys.exit(1)
