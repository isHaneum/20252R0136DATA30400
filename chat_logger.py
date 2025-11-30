import os
from datetime import datetime

class ChatLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.chat_history = []  # Store all messages in memory

    def log_message(self, sender, message):
        """
        Logs a message to the log file and stores it in memory.

        Args:
            sender (str): The sender of the message (e.g., "User" or "Assistant").
            message (str): The message content.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {sender}: {message}"
        self.chat_history.append(log_entry)

    def save_log(self):
        """
        Saves the entire chat history to the log file.
        """
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.chat_history) + "\n")

# Example usage
if __name__ == "__main__":
    logger = ChatLogger()
    logger.log_message("User", "실행시점까지 나눈 모든 대화를 저장하게 해줘")
    logger.log_message("Assistant", "모든 대화를 저장하도록 기능을 확장했습니다.")
    logger.save_log()