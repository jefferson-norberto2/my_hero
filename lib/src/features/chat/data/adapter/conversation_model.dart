import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';

class Conversation {
  final String id;
  String title;
  List<ChatMessage> messages;
  String modelId;
  PreferredBackend preferredBackend;
  int maxTokens;
  bool enableThinking;
  ResponseStyle responseStyle;
  final DateTime createdAt;
  DateTime updatedAt;

  Conversation({
    required this.id,
    required this.title,
    required this.messages,
    String? modelId,
    bool? is2B,
    PreferredBackend? preferredBackend,
    bool? useCpu,
    int? maxTokens,
    this.enableThinking = false,
    this.responseStyle = ResponseStyle.simple,
    required this.createdAt,
    required this.updatedAt,
  })  : modelId = _resolveModelId(modelId: modelId, is2B: is2B),
        preferredBackend = preferredBackend ??
            ((useCpu ?? true) ? PreferredBackend.cpu : PreferredBackend.gpu),
        maxTokens = AppConstants.clampMaxTokens(
          maxTokens ?? AppConstants.defaultMaxTokensForPlatform(),
          modelId: _resolveModelId(modelId: modelId, is2B: is2B),
        );

  static String _resolveModelId({String? modelId, bool? is2B}) {
    if (modelId != null && modelId.isNotEmpty) {
      return AppConstants.modelSpecById(modelId).id;
    }
    return AppConstants.modelSpec(is2B: is2B ?? true).id;
  }

  bool get is2B => modelId == AppConstants.model2b.id;

  set is2B(bool value) {
    modelId = AppConstants.modelSpec(is2B: value).id;
  }

  AiModelSpec get modelSpec => AppConstants.modelSpecById(modelId);

  bool get useCpu => preferredBackend == PreferredBackend.cpu;

  Conversation copyWith({
    String? id,
    String? title,
    List<ChatMessage>? messages,
    String? modelId,
    bool? is2B,
    PreferredBackend? preferredBackend,
    int? maxTokens,
    bool? enableThinking,
    ResponseStyle? responseStyle,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    final resolvedModelId = modelId ??
        (is2B != null ? AppConstants.modelSpec(is2B: is2B).id : this.modelId);
    return Conversation(
      id: id ?? this.id,
      title: title ?? this.title,
      messages: messages ?? List.from(this.messages),
      modelId: resolvedModelId,
      preferredBackend: preferredBackend ?? this.preferredBackend,
      maxTokens: maxTokens ?? this.maxTokens,
      enableThinking: enableThinking ?? this.enableThinking,
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
      'modelId': modelId,
      'is2B': is2B,
      'preferredBackend': preferredBackend.name,
      'useCpu': useCpu,
      'maxTokens': maxTokens,
      'enableThinking': enableThinking,
      'responseStyle': responseStyle.name,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }

  static PreferredBackend _parseBackend(Map<String, dynamic> json) {
    final raw = json['preferredBackend'] as String?;
    if (raw != null) {
      for (final value in PreferredBackend.values) {
        if (value.name == raw) return value;
      }
    }
    final legacyUseCpu = json['useCpu'] as bool? ?? true;
    return legacyUseCpu ? PreferredBackend.cpu : PreferredBackend.gpu;
  }

  factory Conversation.fromJson(Map<String, dynamic> json) {
    final rawModelId = json['modelId'] as String?;
    final legacyIs2B = json['is2B'] as bool?;
    final rawMaxTokens = json['maxTokens'] as int?;
    return Conversation(
      id: json['id'] as String,
      title: json['title'] as String? ?? 'Conversa sem título',
      messages: (json['messages'] as List<dynamic>?)
              ?.map((m) => ChatMessage.fromJson(m as Map<String, dynamic>))
              .toList() ??
          [],
      modelId: rawModelId,
      is2B: legacyIs2B,
      preferredBackend: _parseBackend(json),
      maxTokens: rawMaxTokens,
      enableThinking: json['enableThinking'] as bool? ?? false,
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
