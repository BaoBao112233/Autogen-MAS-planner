import json
import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from template.configs.environments import env


class RedisSupportChatHistory(BaseChatMessageHistory):
    """Chat message history that supports text+image and stores messages in Redis."""

    def __init__(self, session_id: str, conversation_id: str, ttl: int = 3600):
        super().__init__()
        self.redis_client = redis.Redis(
            host=env.REDIS_HOST,
            port=env.REDIS_PORT,
            db=env.REDIS_DB,
            decode_responses=True,
        )
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.ttl = ttl
        self.messages = []
        self._load_messages()

    # -------------------------------
    # Chat history methods
    # -------------------------------
    def _get_redis_key(self):
        return f"chat_history:{self.session_id}:{self.conversation_id}"

    def _load_messages(self):
        """Load messages from Redis."""
        try:
            raw = self.redis_client.get(self._get_redis_key())
            if raw:
                messages_data = json.loads(raw)
                self.messages = []
                for msg in messages_data:
                    if msg["type"] == "human":
                        self.messages.append(HumanMessage(content=msg["data"]["content"]))
                    elif msg["type"] == "ai":
                        self.messages.append(AIMessage(content=msg["data"]["content"]))
                    elif msg["type"] == "system":
                        self.messages.append(SystemMessage(content=msg["data"]["content"]))
            else:
                self.messages = []
        except Exception as e:
            print(f"Error loading messages from Redis: {e}")
            self.messages = []

    def _save_messages(self):
        """Save messages to Redis with TTL."""
        messages_json = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                messages_json.append({"type": "human", "data": {"content": msg.content}})
            elif isinstance(msg, AIMessage):
                messages_json.append({"type": "ai", "data": {"content": msg.content}})
            elif isinstance(msg, SystemMessage):
                messages_json.append({"type": "system", "data": {"content": msg.content}})

        try:
            self.redis_client.setex(
                self._get_redis_key(), self.ttl, json.dumps(messages_json)
            )
        except Exception as e:
            print(f"Error saving messages to Redis: {e}")

    def add_message(self, message):
        """Add a generic message to the store."""
        self.messages.append(message)
        self._save_messages()

    def add_user_message(self, message):
        """Add a user message, optionally with an image."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message):
        """Add an AI message."""
        self.add_message(AIMessage(content=message))

    def clear(self):
        """Clear all messages in Redis."""
        self.messages = []
        try:
            self.redis_client.delete(self._get_redis_key())
        except Exception as e:
            print(f"Error clearing Redis: {e}")

    # -------------------------------
    # Extra session utilities
    # -------------------------------
    def save_session(self, data: dict, ttl: int = 3600) -> bool:
        """Save arbitrary session data with TTL."""
        key = f"session:{self.session_id}"
        try:
            self.redis_client.setex(key, ttl, json.dumps(data))
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def get_session(self):
        """Retrieve arbitrary session data."""
        key = f"session:{self.session_id}"
        try:
            raw = self.redis_client.get(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            print(f"Error getting session: {e}")
        return None

    def increment_counter(self) -> int | None:
        """Increment counter per session."""
        key = f"counter:{self.session_id}"
        try:
            return self.redis_client.incr(key)
        except Exception as e:
            print(f"Error incrementing counter: {e}")
            return None

    def exists_session(self) -> bool:
        """Check if session exists."""
        key = f"session:{self.session_id}"
        try:
            return self.redis_client.exists(key) == 1
        except Exception as e:
            print(f"Error checking session existence: {e}")
            return False


# --- Demo ---
if __name__ == "__main__":
    sid = "12345"
    cid = "conv1"
    chat_history = RedisSupportChatHistory(sid, cid)

    chat_history.add_user_message("Xin chào!", image_url=None)
    chat_history.add_ai_message("Chào bạn, tôi là AI Agent.")
    print("Messages:", chat_history.messages)

    print("Save session:", chat_history.save_session({"user": "baobao"}, ttl=20))
    print("Get session:", chat_history.get_session())
    print("Exists session:", chat_history.exists_session())
    print("Counter 1:", chat_history.increment_counter())
    print("Counter 2:", chat_history.increment_counter())
