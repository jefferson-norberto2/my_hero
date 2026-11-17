import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';

void main() {
  group('buildContextPresets', () {
    test('returns only [4096] when maxContextLength is <= 4096', () {
      expect(buildContextPresets(0), [4096]);
      expect(buildContextPresets(2048), [4096]);
      expect(buildContextPresets(4096), [4096]);
    });

    test('generates exact powers of 2 from 4096 up to maxContextLength', () {
      expect(buildContextPresets(8192), [4096, 8192]);
      expect(buildContextPresets(16384), [4096, 8192, 16384]);
      expect(buildContextPresets(32768), [4096, 8192, 16384, 32768]);
      expect(
        buildContextPresets(262144),
        [4096, 8192, 16384, 32768, 65536, 131072, 262144],
      );
      expect(
        buildContextPresets(1048576),
        [
          4096,
          8192,
          16384,
          32768,
          65536,
          131072,
          262144,
          524288,
          1048576,
        ],
      );
    });

    test('appends non-power-of-two maxContextLength as final item', () {
      expect(buildContextPresets(5000), [4096, 5000]);
      expect(buildContextPresets(40000), [4096, 8192, 16384, 32768, 40000]);
      expect(buildContextPresets(40960), [4096, 8192, 16384, 32768, 40960]);
    });
  });

  group('formatContextLabel', () {
    test('formats exact megabytes, kilobytes, and decimal intermediate values', () {
      expect(formatContextLabel(4096), '4k');
      expect(formatContextLabel(8192), '8k');
      expect(formatContextLabel(16384), '16k');
      expect(formatContextLabel(32768), '32k');
      expect(formatContextLabel(65536), '64k');
      expect(formatContextLabel(131072), '128k');
      expect(formatContextLabel(262144), '256k');
      expect(formatContextLabel(524288), '512k');
      expect(formatContextLabel(1048576), '1M');
      expect(formatContextLabel(2097152), '2M');
      expect(formatContextLabel(40000), '39.1k');
      expect(formatContextLabel(5000), '4.9k');
    });
  });

  group('resolveClosestContextPreset', () {
    test('preserves valid steps and snaps out-of-range or intermediate values to closest step without exceeding max', () {
      const presets32k = [4096, 8192, 16384, 32768];
      const presets50k = [4096, 8192, 16384, 32768, 50000];

      expect(resolveClosestContextPreset(16384, presets32k), 16384);
      expect(resolveClosestContextPreset(65536, presets32k), 32768);
      expect(resolveClosestContextPreset(40000, presets32k), 32768);
      expect(resolveClosestContextPreset(12000, presets32k), 8192);
      expect(resolveClosestContextPreset(45000, presets50k), 50000);
      expect(resolveClosestContextPreset(40000, presets50k), 32768);
      expect(resolveClosestContextPreset(2048, presets32k), 4096);
    });
  });
}
