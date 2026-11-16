/// Defines the verbosity and tone of assistant responses.
enum ResponseStyle {
  brief,
  simple,
  detailed;

  String get label {
    switch (this) {
      case ResponseStyle.brief:
        return 'Curta';
      case ResponseStyle.simple:
        return 'Padrão';
      case ResponseStyle.detailed:
        return 'Longa';
    }
  }

  String get description {
    switch (this) {
      case ResponseStyle.brief:
        return 'Respostas curtas, objetivas e diretas ao ponto.';
      case ResponseStyle.simple:
        return 'Respostas dadas de acordo com o treinamento do modelo.';
      case ResponseStyle.detailed:
        return 'Respostas aprofundadas, explicativas e com mais detalhes.';
    }
  }

  /// Instruction appended to the end of the user prompt to guide the AI response.
  String get promptInstruction {
    switch (this) {
      case ResponseStyle.brief:
        return 'Seja breve e direto na resposta.';
      case ResponseStyle.simple:
        return '';
      case ResponseStyle.detailed:
        return 'Dê uma resposta detalhada, explicativa e completa, com exemplos se possível.';
    }
  }

  /// Appends the style instruction to the prompt sent to the model.
  String applyToPrompt(String prompt) {
    final trimmed = prompt.trim();
    if (trimmed.isEmpty) return trimmed;
    return '$trimmed\n\n$promptInstruction';
  }

  static ResponseStyle fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'brief':
      case 'breve':
        return ResponseStyle.brief;
      case 'detailed':
      case 'detalhada':
        return ResponseStyle.detailed;
      case 'simple':
      case 'simples':
      default:
        return ResponseStyle.simple;
    }
  }
}
