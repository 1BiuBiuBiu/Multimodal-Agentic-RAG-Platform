import unittest
import time
from app.chat_worker import SessionTask, get_active_task, remove_task, _REGISTRY

class TestChatWorker(unittest.TestCase):
    def setUp(self):
        _REGISTRY.clear()

    def test_session_task_lifecycle(self):
        task = SessionTask("sess-test-1", "test prompt", {}, [], "http://127.0.0.1:8000")
        self.assertEqual(task.status, "running")
        self.assertEqual(task.prompt, "test prompt")
        
        _REGISTRY["sess-test-1"] = task
        retrieved = get_active_task("sess-test-1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, "sess-test-1")
        
        # Test fragment locking and updates
        with task.lock:
            task.fragments.append("Hello ")
            task.fragments.append("world")
            task.status = "done"
            
        self.assertEqual("".join(task.fragments), "Hello world")
        self.assertEqual(task.status, "done")
        
        # Test removal
        remove_task("sess-test-1")
        self.assertIsNone(get_active_task("sess-test-1"))

if __name__ == "__main__":
    unittest.main()
