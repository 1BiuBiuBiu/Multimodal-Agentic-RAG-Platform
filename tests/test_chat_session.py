import unittest
import time
from datetime import datetime, timedelta
from app.chat_session import (
    create_session,
    save_session,
    get_session_messages,
    delete_session,
    list_sessions,
    group_sessions_by_date,
    _derive_title,
)

class TestChatSession(unittest.TestCase):
    def test_session_lifecycle(self):
        # 1. Create
        sess = create_session("新对话")
        sid = sess["id"]
        self.assertTrue(sid.startswith("sess_"))

        # 2. Save messages & auto-derive title
        test_msgs = [
            {"role": "user", "content": "请介绍一下AffectGPT模型的核心架构与设计思路？"},
            {"role": "assistant", "content": "AffectGPT 采用 Q-Former 结构..."},
        ]
        updated = save_session(sid, test_msgs)
        self.assertEqual(updated["message_count"], 2)
        self.assertIn("请介绍一下AffectGPT", updated["title"])

        # 3. Reload messages from disk
        loaded = get_session_messages(sid)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["content"], test_msgs[0]["content"])

        # 4. Grouping by date
        grouped = group_sessions_by_date(list_sessions())
        self.assertIn("今天", grouped)
        found = any(s["id"] == sid for s in grouped["今天"])
        self.assertTrue(found)

        # 5. Delete session
        delete_session(sid)
        rem = [s for s in list_sessions() if s["id"] == sid]
        self.assertEqual(len(rem), 0)
        self.assertEqual(get_session_messages(sid), [])


if __name__ == "__main__":
    unittest.main()
