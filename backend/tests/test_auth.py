import unittest
from app.auth import get_current_user, require_admin, ADMIN_EMAIL

class TestAuth(unittest.TestCase):
    def test_default_dev_user(self):
        user = get_current_user(None)
        self.assertIn("role", user)
        self.assertEqual(user["role"], "admin")

    def test_admin_check(self):
        admin_user = {"uid": "123", "email": ADMIN_EMAIL, "role": "admin"}
        res = require_admin(admin_user)
        self.assertEqual(res["role"], "admin")

    def test_viewer_denied(self):
        viewer_user = {"uid": "456", "email": "viewer@hospital.org", "role": "viewer"}
        with self.assertRaises(Exception):
            require_admin(viewer_user)

if __name__ == "__main__":
    unittest.main()
