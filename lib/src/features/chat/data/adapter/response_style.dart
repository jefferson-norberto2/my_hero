enum ResponseStyle {
  brief,
  simple,
  detailed;

  String get label {
    switch (this) {
      case ResponseStyle.brief:
        return 'Breve';
      case ResponseStyle.simple:
        return 'Simples';
      case ResponseStyle.detailed:
        return 'Detalhada';
    }
  }

  String get description {
    switch (this) {
      case ResponseStyle.brief:
        return 'Respostas curtas, objetivas e diretas ao ponto.';
      case ResponseStyle.simple:
        return 'Linguagem clara, equilibrada e de fácil entendimento.';
      case ResponseStyle.detailed:
        return 'Respostas aprofundadas, explicativas e com mais detalhes.';
    }
  }

  /// Frase concatenada no final do prompt do usuário para orientar a resposta da IA
  String get promptInstruction {
    switch (this) {
      case ResponseStyle.brief:
        return 'Por favor, seja breve e direto na resposta.';
      case ResponseStyle.simple:
        return 'Por favor, me dê alguns detalhes, mas não precisa se aprofundar.';
      case ResponseStyle.detailed:
        return 'Por favor, dê mais detalhes sobre o assunto.';
    }
  }

  /// Concatena a instrução de estilo ao final do texto enviado para a IA
  String applyToPrompt(String prompt) {
    final trimmed = prompt.trim();
    if (trimmed.isEmpty) return trimmed;
    return '$trimmed\n\n$promptInstruction';
  }

  String get systemInstruction {
    switch (this) {
      case ResponseStyle.brief:
        return 'Você é um assistente prestativo. Responda SEMPRE em português. Suas respostas devem ser extremamente BREVES, diretas e objetivas, sem rodeios ou explicações prolixas.';
      case ResponseStyle.simple:
        return 'Você é um assistente prestativo. Responda SEMPRE em português. Suas respostas devem usar linguagem SIMPLES, amigável, clara e de tamanho equilibrado, evitando complexidade desnecessária.';
      case ResponseStyle.detailed:
        return 'Você é um assistente prestativo. Responda SEMPRE em português. Suas respostas devem ser DETALHADAS, completas, aprofundadas e didáticas, explicando conceitos com riqueza de detalhes e exemplos.';
    }
  }

  static ResponseStyle fromString(String? value) {
    switch (value) {
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
