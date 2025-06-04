from typing import Any, Dict, List, Optional


class Message:
    """
    A class to represent a turn in the LLM conversation.
    This class is used to store the content of the message, its role (OpenAI compatible), and any additional metadata.
    """
    def __init__(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None, tool_calls: Optional[List[Dict[str, Any]]] = [], tool_call_id: Optional[str] = None, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls if tool_calls is not None else []
        self.tool_call_id = tool_call_id if tool_call_id is not None else None
        self.name = name if name is not None else None
        self.metadata = metadata if metadata is not None else {}
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation of the message."""
        return f"Message(role={self.role}, content={self.content})"
    
    # Get message dict without metadata
    def to_message(self) -> Dict[str, Any]:
        """Convert the message to a dictionary format without metadata."""
        message = {}
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        if self.name:
            message["name"] = self.name
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls        
        message["content"] = self.content
        message["role"] = self.role
        return message
    
class Messages:
    """
    A class to represent the memory of the LLM conversation.
    This class is used to store the history of messages exchanged in the conversation.
    """
    def __init__(self):
        self.messages = []  # List to store messages

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an assistant message to the memory."""
        message = Message(role="assistant", content=content, metadata=metadata, tool_calls=None, tool_call_id=None, name=None)
        self.messages.append(message)

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message to the memory."""
        message = Message(role="user", content=content, metadata=metadata, tool_calls=None, tool_call_id=None, name=None)
        self.messages.append(message)

    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a system message to the memory."""
        message = Message(role="system", content=content, metadata=metadata, tool_calls=None, tool_call_id=None, name=None)
        self.messages.append(message)

    # def add_tool_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    #     """Add a tool message to the memory."""
    #     message = Message(role="tool", content=content, metadata=metadata)
    #     self.messages.append(message)        

    def get_messages(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all messages in the memory without metadata. If limit is provided, it will return the last n messages.
        If the first message is a system message, it will always be included.
        """
        if limit is not None:
            messages = self.messages[-limit:]
            # Ensure the first message is included if it's a system message
            if self.messages and self.messages[0].role == "system" and self.messages[0] not in messages:
                messages.insert(0, self.messages[0])
            return [message.to_message() for message in messages]
        else:
            return [message.to_message() for message in self.messages]       
         
    def get_detailed_messages(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all messages in the memory with metadata. If limit is provided, it will return the last n messages.
        If the first message is a system message, it will always be included.
        """
        if limit is not None:
            messages = self.messages[-limit:]
            # Ensure the first message is included if it's a system message
            if self.messages and self.messages[0].role == "system" and self.messages[0] not in messages:
                messages.insert(0, self.messages[0])
            return [message.to_dict() for message in messages]
        else:
            return [message.to_dict() for message in self.messages]
        
    # def get_messages_with_tool_calls(self, limit: int = None) -> List[Dict[str, Any]]:
    #     """Get all messages in the memory with tool calls. If limit is provided, it will return the last n messages.
    #     If the first message is a system message, it will always be included.
    #     """
    #     if limit is not None:
    #         messages = self.messages[-limit:]
    #         # Ensure the first message is included if it's a system message
    #         if self.messages and self.messages[0].role == "system" and self.messages[0] not in messages:
    #             messages.insert(0, self.messages[0])
    #         return [message.to_dict() for message in messages]
    #     else:
    #         return [message.to_dict() for message in self.messages]

    def clear(self) -> None:
        """Clear all messages from the memory."""
        self.messages.clear()

    def __len__(self) -> int:
        """Get the number of messages in the memory."""
        return len(self.messages)
    
    def __repr__(self) -> str:
        """String representation of the memory."""
        return f"Memory(messages={self.messages})"
    
    def copy(self) -> 'Messages':
        """Create a copy of the memory."""
        new_memory = Messages()
        new_memory.messages = self.messages.copy()
        return new_memory
    
    def pop(self) -> Optional[Message]:
        """Remove and return the last message from the memory."""
        if self.messages:
            return self.messages.pop()
        return None
    
    def add_message(self, message: Message) -> None:
        """Append another message to the messages."""
        self.messages.append(message)