import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_gemma/flutter_gemma.dart';

sealed class ChatStreamChunk {
  const ChatStreamChunk();
}

class ChatTextChunk extends ChatStreamChunk {
  final String token;

  const ChatTextChunk(this.token);
}

class ChatThinkingChunk extends ChatStreamChunk {
  final String content;

  const ChatThinkingChunk(this.content);
}

class ChatMessage {
  final String text;
  final bool isUser;
  final Uint8List? imageBytes;
  final String? thinkingText;

  ChatMessage({
    required this.text,
    required this.isUser,
    this.imageBytes,
    this.thinkingText,
  });

  bool get hasThinking =>
      thinkingText != null && thinkingText!.trim().isNotEmpty;

  ChatMessage copyWith({
    String? text,
    bool? isUser,
    Uint8List? imageBytes,
    String? thinkingText,
  }) {
    return ChatMessage(
      text: text ?? this.text,
      isUser: isUser ?? this.isUser,
      imageBytes: imageBytes ?? this.imageBytes,
      thinkingText: thinkingText ?? this.thinkingText,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'isUser': isUser,
      'imageBytes': imageBytes != null ? base64Encode(imageBytes!) : null,
      if (thinkingText != null) 'thinkingText': thinkingText,
    };
  }

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(
      text: json['text'] as String? ?? '',
      isUser: json['isUser'] as bool? ?? false,
      imageBytes: json['imageBytes'] != null
          ? base64Decode(json['imageBytes'] as String)
          : null,
      thinkingText: json['thinkingText'] as String?,
    );
  }
}

class ChatMessageAdapter {
  static Message toGemmaMessage(ChatMessage message) {
    if (message.imageBytes != null) {
      return Message.withImage(
        text: message.text,
        imageBytes: message.imageBytes!,
        isUser: message.isUser,
      );
    }
    return Message.text(text: message.text, isUser: message.isUser);
  }
}
