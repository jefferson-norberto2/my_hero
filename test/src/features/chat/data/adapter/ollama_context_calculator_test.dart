import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_context_calculator.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';

void main() {
  group('OllamaContextCalculator.parseMaxContextFromShowResponse', () {
    test('extracts .context_length from different model families', () {
      expect(
        OllamaContextCalculator.parseMaxContextFromShowResponse({
          'model_info': {'llama.context_length': 131072},
        }),
        131072,
      );

      expect(
        OllamaContextCalculator.parseMaxContextFromShowResponse({
          'model_info': {
            'general.architecture': 'qwen2',
            'qwen2.context_length': 32768,
          },
        }),
        32768,
      );
    });

    test('falls back to 4096 when model_info or context_length is missing or invalid', () {
      expect(
        OllamaContextCalculator.parseMaxContextFromShowResponse(null),
        4096,
      );
      expect(
        OllamaContextCalculator.parseMaxContextFromShowResponse({
          'model_info': {'general.architecture': 'llama'},
        }),
        4096,
      );
      expect(
        OllamaContextCalculator.parseMaxContextFromShowResponse({
          'model_info': {'llama.context_length': 0},
        }),
        4096,
      );
    });
  });

  group('OllamaContextCalculator.resolveNumCtx', () {
    test('stays at current fixed step below 80% threshold and scales up at or above 80%', () {
      final belowThreshold = OllamaContextCalculator.resolveNumCtx(
        mode: ContextMode.automatic,
        manualContextLength: 4096,
        modelMaxContext: 131072,
        currentStepNumCtx: 4096,
        estimatedTokens: 3276, // 79.98% of 4096
      );
      expect(belowThreshold, 4096);

      final atThreshold = OllamaContextCalculator.resolveNumCtx(
        mode: ContextMode.automatic,
        manualContextLength: 4096,
        modelMaxContext: 131072,
        currentStepNumCtx: 4096,
        estimatedTokens: 3277, // 80.00% of 4096
      );
      expect(atThreshold, 8192);

      // Large jump advances through multiple fixed steps at once.
      final multiJump = OllamaContextCalculator.resolveNumCtx(
        mode: ContextMode.automatic,
        manualContextLength: 4096,
        modelMaxContext: 131072,
        currentStepNumCtx: 4096,
        estimatedTokens: 15000, // >= 80% of 4096, 8192, and 16384 -> steps to 32768
      );
      expect(multiJump, 32768);
    });

    test('never shrinks current step mid-chat and caps automatic mode strictly at modelMaxContext', () {
      final keptStep = OllamaContextCalculator.resolveNumCtx(
        mode: ContextMode.automatic,
        manualContextLength: 4096,
        modelMaxContext: 65536,
        currentStepNumCtx: 16384,
        estimatedTokens: 500,
      );
      expect(keptStep, 16384);

      final cappedByModel = OllamaContextCalculator.resolveNumCtx(
        mode: ContextMode.automatic,
        manualContextLength: 4096,
        modelMaxContext: 16384,
        currentStepNumCtx: 4096,
        estimatedTokens: 20000,
      );
      expect(cappedByModel, 16384);
    });

    test('manual mode returns selected preset capped to modelMaxContext', () {
      expect(
        OllamaContextCalculator.resolveNumCtx(
          mode: ContextMode.manual,
          manualContextLength: 32768,
          modelMaxContext: 16384,
          currentStepNumCtx: 4096,
          estimatedTokens: 100,
        ),
        16384,
      );

      expect(
        OllamaContextCalculator.resolveNumCtx(
          mode: ContextMode.manual,
          manualContextLength: 8192,
          modelMaxContext: 65536,
          currentStepNumCtx: 4096,
          estimatedTokens: 100,
        ),
        8192,
      );

      expect(
        OllamaContextCalculator.resolveNumCtx(
          mode: ContextMode.manual,
          manualContextLength: 24576,
          modelMaxContext: 32768,
          currentStepNumCtx: 4096,
          estimatedTokens: 100,
        ),
        24576,
      );

      expect(
        OllamaContextCalculator.resolveNumCtx(
          mode: ContextMode.manual,
          manualContextLength: 262144,
          modelMaxContext: 262144,
          currentStepNumCtx: 4096,
          estimatedTokens: 100,
        ),
        262144,
      );

      expect(
        OllamaContextCalculator.resolveNumCtx(
          mode: ContextMode.manual,
          manualContextLength: 2048,
          modelMaxContext: 16384,
          currentStepNumCtx: 4096,
          estimatedTokens: 100,
        ),
        4096,
      );
    });

    test('estimates accumulated conversation tokens cleanly with history or metrics', () {
      final fromHistory = OllamaContextCalculator.estimateConversationTokens(
        history: [
          ChatMessage(text: 'a' * 400, isUser: true),
          ChatMessage(text: 'Sistema: Ignorado', isUser: false),
        ],
        newUserText: 'b' * 200,
        newAttachment: ChatAttachment(
          name: 'doc.txt',
          type: AttachmentType.document,
          bytes: Uint8List(0),
          textContent: 'c' * 400,
        ),
      );
      expect(fromHistory, greaterThan(250));

      final fromMetrics = OllamaContextCalculator.estimateConversationTokens(
        history: const [],
        newUserText: 'abcd',
        lastPromptTokens: 3000,
        lastEvalTokens: 400,
      );
      expect(fromMetrics, 3401);
    });
  });
}
