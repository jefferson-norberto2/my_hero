# My Hero 🦸‍♂️

Aplicativo Flutter para interação e execução local de modelos de Inteligência Artificial **Google Gemma** (2B e 4B) utilizando a biblioteca `flutter_gemma`, com suporte a streaming de respostas e envio de imagens.

---

## 🏛️ Arquitetura do Projeto

O projeto segue a abordagem **Feature-First** (baseada em funcionalidades), com todo o código-fonte modularizado dentro do diretório `lib/src/`. A estrutura foi projetada para garantir **baixo acoplamento**, **alta testabilidade** e facilidade de escala.

### Camadas Principais

- **`core/`**: Infraestrutura transversal da aplicação. Contém definições de rede, injeção de dependência, constantes, temas e rotas globais.
- **`shared/`**: Recursos compartilhados entre duas ou mais features, como componentes visuais genéricos, utilitários e serviços do sistema operacional (ex: câmera/galeria).
- **`features/`**: Módulos independentes de negócio (ex: `chat`, `login`, etc.). Cada feature encapsula seus próprios dados, regras de estado e interface visual.

---

## 📁 Estrutura de Diretórios

```text
lib/
├── main.dart                                                  # Ponto de entrada: inicialização de bindings, DI e runApp
└── src/
    ├── app.dart                                               # Configuração do MaterialApp, tema e rotas globais
    │
    ├── core/                                                  # Infraestrutura global
    │   ├── constants/
    │   │   └── app_constants.dart                             # URLs de download dos modelos, nomes de arquivos e parâmetros
    │   ├── di/
    │   │   └── injection.dart                                 # Injeção de dependências centralizada (GetIt)
    │   ├── network/
    │   │   └── dio_client.dart                                # Instância e configuração do Dio (timeouts, interceptors)
    │   ├── routes/
    │   │   └── app_routes.dart                                # Definição de rotas e navegação da aplicação
    │   └── theme/
    │       └── app_theme.dart                                 # Configuração de temas e paleta visual (Material 3)
    │
    ├── shared/                                                # Recursos reutilizáveis entre features
    │   └── services/
    │       └── media_picker_service.dart                      # Abstração para captura de fotos da galeria (ImagePicker)
    │
    └── features/                                              # Módulos funcionais
        └── chat/                                              # Feature de Chat com Gemma
            ├── data/                                          # Camada de Dados
            │   ├── adapter/
            │   │   ├── chat_message_adapter.dart              # Modelos de domínio (ChatMessage) e conversores para o Gemma
            │   │   └── conversation_model.dart                # Modelo de conversa (histórico de mensagens e modelo selecionado)
            │   ├── datasource/
            │   │   ├── chat_history_datasource.dart           # Persistência do histórico de conversas em disco local
            │   │   ├── gemma_local_datasource.dart            # Comunicação de baixo nível com o plugin flutter_gemma
            │   │   └── model_download_datasource.dart         # Download de arquivos com Dio e verificação no disco local
            │   └── repository/
            │       ├── chat_repository.dart                   # Interface/Contrato do repositório
            │       └── chat_repository_impl.dart              # Implementação orquestrando os datasources e contexto
            │
            ├── controller/                                    # Camada de Estado e Regras de Apresentação
            │   └── chat_controller.dart                       # Gerenciamento de estado reativo com Signals e conversas
            │
            └── ui/                                            # Camada de Interface do Usuário
                ├── pages/
                │   └── chat_page.dart                         # Tela principal com menu lateral e topo dinâmico
                └── widgets/
                    ├── chat_bubble.dart                       # Renderização do balão de mensagem (texto e anexo)
                    ├── chat_drawer.dart                       # Menu lateral hamburger com histórico e + Nova Conversa
                    ├── chat_input_bar.dart                    # Barra inferior (input de texto, anexar foto, enviar/parar)
                    ├── chat_message_list.dart                 # Lista de mensagens com rolagem automática
                    ├── model_selector_widget.dart             # Seleção CPU/GPU, botões 2B/4B e barra de progresso
                    └── model_switch_dialog.dart               # Modal para troca dinâmica de modelo com preservação de contexto
```

---

## 🛠️ Tecnologias e Padrões Adotados

### 1. Injeção de Dependências (`get_it`)
Localizada em `lib/src/core/di/injection.dart`.
- **Singletons:** Serviços e fontes de dados com ciclo de vida único durante toda a execução da aplicação (`DioClient`, `MediaPickerService`, `GemmaLocalDatasource`, `ModelDownloadDatasource`, `ChatRepository`).
- **Factories:** Instâncias criadas sob demanda por tela/ciclo de vida (`ChatController`).

### 2. Gerenciamento de Estado Reativo (`signals_flutter`)
Os estados das features utilizam o conceito de **Signals** para obter reatividade granular e eficiente:
- **`signal(valor)`**: Criação de variáveis reativas no Controller (`isModelLoaded`, `isLoading`, `isGenerating`, `downloadProgress`, `chatHistory`, etc.).
- **`SignalBuilder`**: Construtor declarativo na UI que reconstrói exclusivamente o widget específico quando um sinal lido em seu escopo é alterado, eliminando rebuilds desnecessários da árvore inteira.
- **`effect(() { ... })`**: Disparo de efeitos colaterais como alertas visuais (`SnackBar`) e mensagens de erro sem poluir o fluxo de build.

### 3. Separação de Dados: Adapter, Datasource e Repository
- **Adapter:** Converte entidades externas/DTOs (como `Message` do plugin Gemma) para modelos de domínio desacoplados do app (`ChatMessage`).
- **Datasource:** Acessa uma única fonte de dados de forma crua (rede, banco local, plugin nativo).
- **Repository:** Implementa o contrato de domínio, tratando erros e combinando múltiplos datasources se necessário.

### 4. Histórico de Conversas e Troca de Modelo com Contexto
- **Menu Lateral Hamburger (`ChatDrawer`):** Permite alternar entre conversas salvas, criar novas conversas (`+ Nova Conversa`) ou excluir sessões anteriores.
- **Persistência Local (`ChatHistoryDatasource`):** Armazena todas as conversas e mensagens no armazenamento do dispositivo em formato JSON offline.
- **Preservação de Contexto ao Trocar de Modelo (`ModelSwitchDialog`):** Ao alternar entre Gemma 2B e 4B (ou entre CPU e GPU) dentro de uma conversa, a engine carrega o novo modelo e sincroniza o histórico recente de mensagens via `addQueryChunk`, permitindo que o novo modelo continue respondendo com base no que já foi conversado.

---

## 🚀 Como Adicionar uma Nova Feature

Para criar uma nova funcionalidade (por exemplo, `login` ou `history`), siga o padrão:

1. Crie a pasta da feature em `lib/src/features/<nome_da_feature>/`:
   ```text
   features/<nome_da_feature>/
   ├── data/
   │   ├── adapter/
   │   ├── datasource/
   │   └── repository/
   ├── controller/
   └── ui/
       ├── pages/
       └── widgets/
   ```
2. Defina os contratos e implementações em `data/`.
3. Crie o Controller gerenciando os estados através de `signals`.
4. Registre o novo Datasource, Repository e Controller em `lib/src/core/di/injection.dart`.
5. Construa os widgets e telas em `ui/` consumindo o Controller via `SignalBuilder`.
6. Adicione a rota correspondente em `lib/src/core/routes/app_routes.dart`.

---

## 💻 Como Executar

1. **Obter dependências:**
   ```bash
   flutter pub get
   ```

2. **Verificar análise estática:**
   ```bash
   flutter analyze
   ```

3. **Executar testes unitários:**
   ```bash
   flutter test
   ```

4. **Executar a aplicação:**
   ```bash
   flutter run
   ```
