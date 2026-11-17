import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_context_calculator.dart';
import 'package:my_hero/src/shared/models/compute_mode.dart';
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

  group('OllamaContextCalculator.calculateHardwareContextCeiling', () {
    test('returns null when server memory limit is unset or non-positive', () {
      expect(
        OllamaContextCalculator.calculateHardwareContextCeiling(
          serverMemoryLimitGb: null,
          loadedModelSizeBytes: 4 * 1024 * 1024 * 1024,
          computeMode: ComputeMode.gpu,
          modelMaxContext: 131072,
        ),
        isNull,
      );
      expect(
        OllamaContextCalculator.calculateHardwareContextCeiling(
          serverMemoryLimitGb: 0,
          loadedModelSizeBytes: 0,
          computeMode: ComputeMode.cpu,
          modelMaxContext: 32768,
        ),
        isNull,
      );
    });

    test('discounts loaded model size and applies stricter margin for CPU than Hybrid and GPU', () {
      const limitGb = 12.0;
      const loadedBytes = 6 * 1024 * 1024 * 1024;

      final cpuCeiling = OllamaContextCalculator.calculateHardwareContextCeiling(
        serverMemoryLimitGb: limitGb,
        loadedModelSizeBytes: loadedBytes,
        computeMode: ComputeMode.cpu,
        modelMaxContext: 131072,
      )!;
      final hybridCeiling = OllamaContextCalculator.calculateHardwareContextCeiling(
        serverMemoryLimitGb: limitGb,
        loadedModelSizeBytes: loadedBytes,
        computeMode: ComputeMode.hybrid,
        modelMaxContext: 131072,
      )!;
      final gpuCeiling = OllamaContextCalculator.calculateHardwareContextCeiling(
        serverMemoryLimitGb: limitGb,
        loadedModelSizeBytes: loadedBytes,
        computeMode: ComputeMode.gpu,
        modelMaxContext: 131072,
      )!;

      // 6 GB remaining -> 24,576 raw tokens @ 256 KB/token; CPU uses 50% (12,288),
      // Hybrid uses 65% (15,974), GPU uses 80% (19,660).
      expect(cpuCeiling, 12288);
      expect(hybridCeiling, 15974);
      expect(gpuCeiling, 19660);
      expect(cpuCeiling, lessThan(hybridCeiling));
      expect(hybridCeiling, lessThan(gpuCeiling));
    });

    test('clamps ceiling between fallback 4096 floor and modelMaxContext', () {
      final lowHeadroomCeiling = OllamaContextCalculator.calculateHardwareContextCeiling(
        serverMemoryLimitGb: 8.0,
        loadedModelSizeBytes: 8 * 1024 * 1024 * 1024,
        computeMode: ComputeMode.cpu,
        modelMaxContext: 32768,
      );
      expect(lowHeadroomCeiling, 4096);

      final highHeadroomCeiling = OllamaContextCalculator.calculateHardwareContextCeiling(
        serverMemoryLimitGb: 128.0,
        loadedModelSizeBytes: 2 * 1024 * 1024 * 1024,
        computeMode: ComputeMode.gpu,
        modelMaxContext: 16384,
      );
      expect(highHeadroomCeiling, 16384);
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

    test('never shrinks current step mid-chat and caps automatic mode at hardware ceiling', () {
      final keptStep = OllamaContextCalculator.resolveNumCtx(
        mode: ContextMode.automatic,
        manualContextLength: 4096,
        modelMaxContext: 65536,
        currentStepNumCtx: 16384,
        estimatedTokens: 500,
      );
      expect(keptStep, 16384);

      // 10 GB server with 6 GB model on CPU => 4 GB free * 0.50 = 2 GB => 8192 hardware ceiling.
      final cappedByHardware = OllamaContextCalculator.resolveNumCtx(
        mode: ContextMode.automatic,
        manualContextLength: 4096,
        modelMaxContext: 131072,
        currentStepNumCtx: 4096,
        estimatedTokens: 20000,
        serverMemoryLimitGb: 10.0,
        loadedModelSizeBytes: 6 * 1024 * 1024 * 1024,
        computeMode: ComputeMode.cpu,
      );
      expect(cappedByHardware, 8192);
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
