import 'dart:math' as math;
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';

/// Parses model context capabilities and computes stepped or manual `num_ctx`.
class OllamaContextCalculator {
  const OllamaContextCalculator._();

  // Multimodal image attachment overhead estimate in tokens.
  static const int _imageTokenEstimate = 512;

  // Approximate system instruction overhead when computing from raw history.
  static const int _systemPromptTokenEstimate = 80;

  /// Extracts `maxContextLength` from a `POST /api/show` response JSON map.
  ///
  /// Scans `model_info` for any key ending with `.context_length`
  /// (for example `llama.context_length` or `qwen2.context_length`),
  /// returning [fallback] when missing or non-positive.
  static int parseMaxContextFromShowResponse(
    Map<String, dynamic>? json, {
    int fallback = ContextSettings.fallbackMaxContext,
  }) {
    if (json == null) return fallback;

    final modelInfo = json['model_info'];
    if (modelInfo is Map<String, dynamic>) {
      for (final entry in modelInfo.entries) {
        if (entry.key.endsWith('.context_length')) {
          final parsed = (entry.value as num?)?.toInt();
          if (parsed != null && parsed > 0) {
            return parsed;
          }
        }
      }
    }

    return fallback;
  }

  /// Estimates total accumulated tokens for the upcoming request turn.
  static int estimateConversationTokens({
    required List<ChatMessage> history,
    required String newUserText,
    ChatAttachment? newAttachment,
    int lastPromptTokens = 0,
    int lastEvalTokens = 0,
  }) {
    final newTurnTokens = _estimateMessageTokens(
      text: newUserText,
      attachment: newAttachment,
      hasImage: newAttachment?.type == AttachmentType.image,
    );

    if (lastPromptTokens > 0 || lastEvalTokens > 0) {
      return lastPromptTokens + lastEvalTokens + newTurnTokens;
    }

    var total = _systemPromptTokenEstimate + newTurnTokens;
    for (final msg in history) {
      if (msg.text.startsWith('Sistema:') || msg.text.trim().isEmpty) {
        continue;
      }
      total += _estimateMessageTokens(
        text: msg.content,
        attachment: msg.attachment,
        hasImage: msg.imageBytes != null,
      );
    }
    return total;
  }

  static int _estimateMessageTokens({
    required String text,
    ChatAttachment? attachment,
    required bool hasImage,
  }) {
    var chars = text.length;
    if (attachment?.textContent != null) {
      chars += attachment!.textContent!.length + 64;
    }
    var tokens = (chars / 4).ceil();
    if (hasImage) {
      tokens += _imageTokenEstimate;
    }
    return tokens;
  }

  /// Resolves the target `num_ctx` value for the next Ollama API request.
  static int resolveNumCtx({
    required ContextMode mode,
    required int manualContextLength,
    required int modelMaxContext,
    required int currentStepNumCtx,
    required int estimatedTokens,
  }) {
    final safeModelMax = modelMaxContext > 0
        ? modelMaxContext
        : ContextSettings.fallbackMaxContext;

    if (mode == ContextMode.manual) {
      if (safeModelMax <= ContextSettings.fallbackMaxContext) {
        return ContextSettings.fallbackMaxContext;
      }
      return math.max(
        ContextSettings.fallbackMaxContext,
        math.min(manualContextLength, safeModelMax),
      );
    }

    final allowedSteps =
        ContextSettings.availablePresetsForModel(safeModelMax);

    // Preserve current step if already valid so num_ctx never shrinks mid-chat.
    var activeStep = allowedSteps.first;
    if (currentStepNumCtx > 0) {
      final matchingCurrent =
          allowedSteps.where((s) => s <= currentStepNumCtx).lastOrNull;
      if (matchingCurrent != null) {
        activeStep = matchingCurrent;
      }
    }

    // Advance cleanly to higher fixed steps only when exceeding the 80% threshold.
    var stepIndex = allowedSteps.indexOf(activeStep);
    while (stepIndex < allowedSteps.length - 1 &&
        estimatedTokens >= activeStep * ContextSettings.stepUpThreshold) {
      stepIndex++;
      activeStep = allowedSteps[stepIndex];
    }

    return activeStep;
  }
}
