import 'chat_message_adapter.dart';
import 'response_style.dart';

class Conversation {
  final String id;
  String title;
  List<ChatMessage> messages;
  String modelName;
  ResponseStyle responseStyle;
  final DateTime createdAt;
  DateTime updatedAt;

  Conversation({
    required this.id,
    required this.title,
    required this.messages,
    this.modelName = '',
    this.responseStyle = ResponseStyle.simple,
    required this.createdAt,
    required this.updatedAt,
  });

  // Retrocompatibilidade
  bool get is2B => modelName.contains('2b') || modelName.contains('3b');
  bool get useCpu => true;

  Conversation copyWith({
    String? id,
    String? title,
    List<ChatMessage>? messages,
    String? modelName,
    ResponseStyle? responseStyle,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return Conversation(
      id: id ?? this.id,
      title: title ?? this.title,
      messages: messages ?? List.from(this.messages),
      modelName: modelName ?? this.modelName,
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
      'modelName': modelName,
      'is2B': is2B,
      'responseStyle': responseStyle.name,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }

  factory Conversation.fromJson(Map<String, dynamic> json) {
    final rawModel = json['modelName'] as String?;
    final legacyIs2B = json['is2B'] as bool? ?? true;
    final fallbackModel = rawModel != null && rawModel.isNotEmpty
        ? rawModel
        : (legacyIs2B ? 'llama3.2:3b' : 'llama3.1:8b');

    return Conversation(
      id: json['id'] as String,
      title: json['title'] as String? ?? 'Conversa sem título',
      messages: (json['messages'] as List<dynamic>?)
              ?.map((m) => ChatMessage.fromJson(m as Map<String, dynamic>))
              .toList() ??
          [],
      modelName: fallbackModel,
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
