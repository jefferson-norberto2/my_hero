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

/// Generates an ascending, deduplicated list of power-of-two `num_ctx` steps
/// starting at `4096` up to [maxContextLength].
///
/// When [maxContextLength] is `<= 4096`, returns `[4096]`.
/// When [maxContextLength] is `> 4096` and not an exact power of two,
/// [maxContextLength] is appended as the final item.
List<int> buildContextPresets(int maxContextLength) {
  const defaultCtx = ContextSettings.fallbackMaxContext;
  if (maxContextLength <= defaultCtx) {
    return const [defaultCtx];
  }

  final presets = <int>[defaultCtx];
  var current = defaultCtx;
  while (current <= maxContextLength ~/ 2) {
    current *= 2;
    presets.add(current);
  }

  if (presets.last != maxContextLength) {
    presets.add(maxContextLength);
  }

  return List<int>.unmodifiable(presets);
}

/// Formats a context window token count into a compact label (`4k`, `39.1k`, `1M`).
String formatContextLabel(int tokens) {
  if (tokens <= 0) return '0';
  if (tokens < 1024) return tokens.toString();
  if (tokens % 1048576 == 0) {
    return '${tokens ~/ 1048576}M';
  }
  if (tokens % 1024 == 0) {
    return '${tokens ~/ 1024}k';
  }
  return '${(tokens / 1024).toStringAsFixed(1)}k';
}

/// Resolves [selectedLength] to the closest valid step in [presets]
/// without exceeding the active model's maximum capacity represented by [presets].
int resolveClosestContextPreset(int selectedLength, List<int> presets) {
  if (presets.isEmpty) return ContextSettings.fallbackMaxContext;
  if (presets.contains(selectedLength)) return selectedLength;

  var closest = presets.first;
  var smallestDiff = (closest - selectedLength).abs();
  for (final preset in presets.skip(1)) {
    final diff = (preset - selectedLength).abs();
    if (diff < smallestDiff) {
      smallestDiff = diff;
      closest = preset;
    }
  }
  return closest;
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
