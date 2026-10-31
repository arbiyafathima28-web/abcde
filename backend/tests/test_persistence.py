import unittest
from app import persistence

class TestPersistence(unittest.TestCase):
    def test_persistence_mode(self):
        mode = persistence.initialize()
        self.assertIn(mode, ["sqlite", "firestore"])

    def test_database_info(self):
        info = persistence.get_database_info()
        self.assertIn("mode", info)
        self.assertIn("project_id", info)

    def test_save_audit_log(self):
        mode = persistence.save_audit_log("TEST_EVENT", {"param": 123}, "12:00:00", "test@test.com")
        self.assertIn(mode, ["sqlite", "firestore"])

if __name__ == "__main__":
    unittest.main()
