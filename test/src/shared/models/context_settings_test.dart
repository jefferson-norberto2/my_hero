import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';

void main() {
  group('buildContextPresets', () {
    test('returns only [4096] when maxContextLength is <= 4096', () {
      expect(buildContextPresets(0), [4096]);
      expect(buildContextPresets(2048), [4096]);
      expect(buildContextPresets(4096), [4096]);
    });

    test('rounds intermediate points to 1024 when maxContextLength <= 16384', () {
      expect(buildContextPresets(8192), [4096, 5120, 6144, 7168, 8192]);
      expect(buildContextPresets(16384), [4096, 7168, 10240, 13312, 16384]);
    });

    test('rounds intermediate points to 4096 when maxContextLength > 16384', () {
      expect(buildContextPresets(32768), [4096, 12288, 20480, 24576, 32768]);
      expect(buildContextPresets(40960), [4096, 12288, 24576, 32768, 40960]);
      expect(buildContextPresets(65536), [4096, 20480, 36864, 49152, 65536]);
      expect(buildContextPresets(131072), [4096, 36864, 69632, 98304, 131072]);
      expect(buildContextPresets(262144), [4096, 69632, 135168, 196608, 262144]);
    });

    test('deduplicates and never exceeds maxContextLength on small or non-multiple intervals', () {
      expect(buildContextPresets(5000), [4096, 5000]);
      expect(buildContextPresets(6000), [4096, 5120, 6000]);
      expect(buildContextPresets(10000), [4096, 5120, 7168, 8192, 10000]);

      for (final maxCtx in [4097, 5000, 6000, 10000, 16385, 24576, 100000]) {
        final presets = buildContextPresets(maxCtx);
        expect(presets.first, 4096);
        expect(presets.last, maxCtx);
        expect(presets.length, lessThanOrEqualTo(5));
        expect(presets.toSet().length, presets.length);

        for (var i = 1; i < presets.length; i++) {
          expect(presets[i], greaterThan(presets[i - 1]));
        }
      }
    });
  });
}
