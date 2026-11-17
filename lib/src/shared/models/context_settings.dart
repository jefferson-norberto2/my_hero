/// Context window sizing strategy used when sending requests to Ollama.
enum ContextMode {
  /// Automatically steps up `num_ctx` in fixed increments as conversation grows.
  automatic,

  /// Uses a user-selected preset capped at the model's maximum capacity.
  manual;

  String get label {
    switch (this) {
      case ContextMode.automatic:
        return 'Automático (Dinâmico)';
      case ContextMode.manual:
        return 'Manual';
    }
  }

  String get description {
    switch (this) {
      case ContextMode.automatic:
        return 'Inicia em 4.096 tokens e escala em degraus fixos conforme a conversa cresce, respeitando a capacidade máxima do modelo ativo.';
      case ContextMode.manual:
        return 'Mantém sempre o limite fixo selecionado abaixo, respeitando a capacidade máxima do modelo ativo.';
    }
  }

  static ContextMode fromString(String? value) {
    return ContextMode.values.firstWhere(
      (e) => e.name == value,
      orElse: () => ContextMode.automatic,
    );
  }
}

/// Generates up to 5 sorted, unique `num_ctx` options for a model supporting
/// [maxContextLength] tokens (`4096` default plus up to 4 proportional steps).
///
/// - When [maxContextLength] `<= 4096`, returns `[4096]`.
/// - When [maxContextLength] `> 4096`, divides `(maxContextLength - 4096)` into
///   4 equal intervals, rounding the 3 intermediate points to the nearest
///   `4096` (when `maxContextLength > 16384`) or `1024` (when
///   `maxContextLength <= 16384`), removing duplicates with a [Set].
List<int> buildContextPresets(int maxContextLength) {
  const defaultCtx = ContextSettings.fallbackMaxContext;
  if (maxContextLength <= defaultCtx) {
    return const [defaultCtx];
  }

  final roundingUnit = maxContextLength > 16384 ? 4096 : 1024;
  final step = (maxContextLength - defaultCtx) / 4;
  final presets = <int>{defaultCtx};

  for (var i = 1; i <= 3; i++) {
    final rawPoint = defaultCtx + (step * i);
    final rounded = (rawPoint / roundingUnit).round() * roundingUnit;
    if (rounded > defaultCtx && rounded < maxContextLength) {
      presets.add(rounded);
    }
  }

  presets.add(maxContextLength);

  final sorted = presets.toList()..sort();
  return List<int>.unmodifiable(sorted);
}

/// Shared constants and formatting helpers for context window configuration.
class ContextSettings {
  const ContextSettings._();

  /// Fallback context window when `/api/show` does not report `.context_length`.
  static const int fallbackMaxContext = 4096;

  /// Fixed context window presets used by both stepped scaling and manual mode.
  static const List<int> stepPresets = [
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
  ];

  /// Threshold fraction (80%) that triggers advancing to the next context step.
  static const double stepUpThreshold = 0.80;

  /// Returns valid presets that do not exceed [maxContextLength].
  static List<int> availablePresetsForModel(int maxContextLength) {
    final safeMax =
        maxContextLength > 0 ? maxContextLength : fallbackMaxContext;
    final filtered = stepPresets.where((p) => p <= safeMax).toList();
    if (filtered.isEmpty) {
      return [safeMax];
    }
    return filtered;
  }

  /// Formats an integer token count using a dot thousands separator (e.g. `8.192`).
  static String formatTokenCount(int tokens) {
    final str = tokens.toString();
    final buffer = StringBuffer();
    for (var i = 0; i < str.length; i++) {
      final remaining = str.length - i;
      if (i > 0 && remaining % 3 == 0) {
        buffer.write('.');
      }
      buffer.write(str[i]);
    }
    return buffer.toString();
  }
}
