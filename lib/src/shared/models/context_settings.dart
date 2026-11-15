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
        return 'Inicia em 4.096 tokens e escala em degraus fixos conforme a conversa cresce, respeitando o limite do modelo e da memória.';
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
