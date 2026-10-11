import 'chat_message_adapter.dart';
import 'response_style.dart';

class Conversation {
  final String id;
  String title;
  List<ChatMessage> messages;
  bool is2B;
  bool useCpu;
  ResponseStyle responseStyle;
  final DateTime createdAt;
  DateTime updatedAt;

  Conversation({
    required this.id,
    required this.title,
    required this.messages,
    this.is2B = true,
    this.useCpu = true,
    this.responseStyle = ResponseStyle.simple,
    required this.createdAt,
    required this.updatedAt,
  });

  Conversation copyWith({
    String? id,
    String? title,
    List<ChatMessage>? messages,
    bool? is2B,
    bool? useCpu,
    ResponseStyle? responseStyle,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return Conversation(
      id: id ?? this.id,
      title: title ?? this.title,
      messages: messages ?? List.from(this.messages),
      is2B: is2B ?? this.is2B,
      useCpu: useCpu ?? this.useCpu,
      responseStyle: responseStyle ?? this.responseStyle,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'messages': messages.map((m) => m.toJson()).toList(),
      'is2B': is2B,
      'useCpu': useCpu,
      'responseStyle': responseStyle.name,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }

  factory Conversation.fromJson(Map<String, dynamic> json) {
    return Conversation(
      id: json['id'] as String,
      title: json['title'] as String? ?? 'Conversa sem título',
      messages: (json['messages'] as List<dynamic>?)
              ?.map((m) => ChatMessage.fromJson(m as Map<String, dynamic>))
              .toList() ??
          [],
      is2B: json['is2B'] as bool? ?? true,
      useCpu: json['useCpu'] as bool? ?? true,
      responseStyle:
          ResponseStyle.fromString(json['responseStyle'] as String?),
      createdAt: json['createdAt'] != null
          ? DateTime.parse(json['createdAt'] as String)
          : DateTime.now(),
      updatedAt: json['updatedAt'] != null
          ? DateTime.parse(json['updatedAt'] as String)
          : DateTime.now(),
    );
  }
}
