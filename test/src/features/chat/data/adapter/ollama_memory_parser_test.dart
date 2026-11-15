import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_memory_parser.dart';
import 'package:my_hero/src/shared/models/compute_mode.dart';

void main() {
  group('OllamaMemoryParser', () {
    test('classifies CPU mode when size_vram == 0 and size > 0', () {
      final allocation = OllamaMemoryParser.parsePsEntry({
        'name': 'llama3.2:latest',
        'size': 5798205849, // ~5.4 GB
        'size_vram': 0,
      });

      expect(allocation.computeMode, ComputeMode.cpu);
      expect(allocation.computeMode.tagLabel, 'CPU');
      expect(allocation.computeMode.isActive, isTrue);
      expect(allocation.displayedBytes, 5798205849);
      expect(allocation.ramBytes, 5798205849);
      expect(allocation.vramBytes, 0);
      expect(allocation.formattedMemory, '5.4 GB');
      expect(allocation.hardwareTooltip, 'Memória RAM (CPU): 5.4 GB');
    });

    test('classifies GPU mode when size_vram > 0 and size_vram >= size', () {
      final exactGpu = OllamaMemoryParser.parsePsEntry({
        'name': 'mimo-v2.6:9b',
        'size': 4080218931, // ~3.8 GB
        'size_vram': 4080218931,
      });

      expect(exactGpu.computeMode, ComputeMode.gpu);
      expect(exactGpu.computeMode.tagLabel, 'GPU');
      expect(exactGpu.displayedBytes, 4080218931);
      expect(exactGpu.vramBytes, 4080218931);
      expect(exactGpu.ramBytes, 0);
      expect(exactGpu.formattedMemory, '3.8 GB');
      expect(exactGpu.hardwareTooltip, 'VRAM: 3.8 GB | RAM: 0.0 GB');

      final greaterVram = OllamaMemoryParser.fromBytes(
        sizeBytes: 4000000000,
        vramBytes: 4080218931,
      );
      expect(greaterVram.computeMode, ComputeMode.gpu);
      expect(greaterVram.displayedBytes, 4080218931);
      expect(greaterVram.ramBytes, 0);
    });

    test('classifies Hybrid mode when 0 < size_vram < size', () {
      const sizeBytes = 5798205849; // ~5.4 GB
      const vramBytes = 3221225472; // 3.0 GB

      final allocation = OllamaMemoryParser.parsePsEntry({
        'name': 'qwen3:8b',
        'size': sizeBytes,
        'size_vram': vramBytes,
      });

      expect(allocation.computeMode, ComputeMode.hybrid);
      expect(allocation.computeMode.tagLabel, 'GPU/CPU');
      expect(allocation.displayedBytes, sizeBytes);
      expect(allocation.vramBytes, vramBytes);
      expect(allocation.ramBytes, sizeBytes - vramBytes);
      expect(allocation.formattedMemory, '5.4 GB');
      expect(allocation.formattedVram, '3.0 GB');
      expect(allocation.formattedRam, '2.4 GB');
      expect(allocation.hardwareTooltip, 'VRAM: 3.0 GB | RAM: 2.4 GB');
    });

    test('returns 0.0 GB without active tag when unloaded (size == 0)', () {
      final allocation = OllamaMemoryParser.parsePsEntry({
        'name': 'llama3.2:latest',
        'size': 0,
        'size_vram': 0,
      });

      expect(allocation.computeMode, ComputeMode.none);
      expect(allocation.computeMode.tagLabel, '');
      expect(allocation.computeMode.isActive, isFalse);
      expect(allocation.displayedBytes, 0);
      expect(allocation.formattedMemory, '0.0 GB');
      expect(allocation.hardwareTooltip, 'Nenhum modelo carregado na memória');

      final fromNull = OllamaMemoryParser.parsePsEntry(null);
      expect(fromNull, allocation);
    });
  });
}
