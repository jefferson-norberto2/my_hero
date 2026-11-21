# My Hero 🦸‍♂️

Aplicativo Flutter multiplataforma (**Android**, **Linux** e **Windows**) para interação e execução local de modelos de Inteligência Artificial abertos do catálogo **LiteRT-LM** (incluindo **Google Gemma 4 E2B/E4B**, **Qwen3**, **DeepSeek R1**, **Phi-4 Mini**, **SmolLM3**, **SmolVLM2**, **LFM2.5**, entre outros) utilizando `flutter_edge_ai` (`2.7.1`) + `flutter_edge_ai_litertlm` (`1.21.0`) + `litertlm_dart` (`0.19.1+1`) via **Dart FFI + Native Assets**, com download sob demanda, gestão de modelos baixados, streaming, **Thinking Mode**, seleção de **janela de contexto (`maxTokens`)**, escolha de backend (**CPU / GPU / NPU**) e envio de imagens.

---

## 🏛️ Arquitetura do Projeto

O projeto segue a abordagem **Feature-First** (baseada em funcionalidades), com todo o código-fonte modularizado dentro do diretório `lib/src/`. A estrutura foi projetada para garantir **baixo acoplamento**, **alta testabilidade** e facilidade de escala.

### Camadas Principais

- **`core/`**: Infraestrutura transversal da aplicação. Contém injeção de dependência, constantes/catálogo de modelos, temas e rotas globais.
- **`shared/`**: Recursos compartilhados entre duas ou mais features, como serviços do sistema operacional (galeria/câmera).
- **`features/`**: Módulos independentes de negócio (ex: `chat`). Cada feature encapsula seus próprios dados, regras de estado e interface visual.

---

## 📁 Estrutura de Diretórios

```text
lib/
├── main.dart                                                  # Ponto de entrada: inicialização de bindings, DI, LiteRtLmEngine e runApp
└── src/
    ├── app.dart                                               # Configuração do MaterialApp, tema e rotas globais
    │
    ├── core/                                                  # Infraestrutura global
    │   ├── constants/
    │   │   └── app_constants.dart                             # Catálogo Gemma 4 (URLs, limites de tokens 1K–32K e suporte por plataforma)
    │   ├── di/
    │   │   └── injection.dart                                 # Injeção de dependências centralizada (GetIt)
    │   ├── routes/
    │   │   └── app_routes.dart                                # Definição de rotas e navegação da aplicação
    │   └── theme/
    │       └── app_theme.dart                                 # Configuração de temas e paleta visual (Material 3)
    │
    ├── shared/                                                # Recursos reutilizáveis entre features
    │   └── services/
    │       ├── inference_notification_service.dart            # Foreground Service Android para inferência em segundo plano + notificação
    │       └── media_picker_service.dart                      # Abstração para seleção de imagens (Galeria / Câmera no mobile)
    │
    └── features/                                              # Módulos funcionais
        └── chat/                                              # Feature de Chat com Gemma 4
            ├── data/                                          # Camada de Dados
            │   ├── adapter/
            │   │   ├── chat_message_adapter.dart              # ChatMessage (texto, imagem, thinkingText), ChatStreamChunk e adapter Gemma
            │   │   ├── conversation_model.dart                # Modelo de conversa (histórico, backend, maxTokens, thinking, estilo)
            │   │   └── response_style.dart                    # Estilos de resposta (Breve, Simples, Detalhada)
            │   ├── datasource/
            │   │   ├── chat_history_datasource.dart           # Persistência do histórico em disco (Application Support no Desktop)
            │   │   └── gemma_local_datasource.dart            # Comunicação com flutter_edge_ai + LiteRtLmEngine (catálogo oficial e FFI)
            │   └── repository/
            │       ├── chat_repository.dart                   # Interface/Contrato do repositório
            │       └── chat_repository_impl.dart              # Implementação orquestrando instalador nativo, inferência e histórico
            │
            ├── controller/                                    # Camada de Estado e Regras de Apresentação
            │   └── chat_controller.dart                       # Gerenciamento de estado reativo com Signals e conversas
            │
            └── ui/                                            # Camada de Interface do Usuário
                ├── pages/
                │   └── chat_page.dart                         # Tela principal com menu lateral e topo dinâmico (backend ativo)
                └── widgets/
                    ├── chat_bubble.dart                       # Balão de mensagem (Markdown, imagem e bloco retrátil de Pensamento)
                    ├── chat_drawer.dart                       # Menu lateral com histórico e + Nova Conversa
                    ├── chat_input_bar.dart                    # Barra inferior (texto, anexar imagem adaptada por plataforma, enviar/parar)
                    ├── chat_message_list.dart                 # Lista de mensagens com rolagem automática
                    ├── model_catalog_list.dart                # Lista dinâmica do catálogo aberto com filtro Baixados, selos e exclusão
                    ├── model_selector_widget.dart             # Seleção do catálogo, CPU/GPU/NPU, maxTokens, Thinking e download nativo
                    └── model_switch_dialog.dart               # Modal para troca dinâmica de modelo do catálogo/backend/contexto/thinking
```

---

## 🛠️ Tecnologias e Padrões Adotados

### 1. Injeção de Dependências (`get_it`)
Localizada em `lib/src/core/di/injection.dart`.
- **Singletons:** Serviços e fontes de dados com ciclo de vida único (`MediaPickerService`, `GemmaLocalDatasource`, `ChatHistoryDatasource`, `ChatRepository`).
- **Factories:** Instâncias criadas sob demanda por tela/ciclo de vida (`ChatController`).

### 2. Gerenciamento de Estado Reativo (`signals_flutter`)
Os estados das features utilizam **Signals** para reatividade granular (`SignalBuilder` e `effect`).

### 3. Engine Modular LiteRT-LM (`flutter_edge_ai 2.7.1` + `flutter_edge_ai_litertlm 1.21.0`)
- **Dart FFI + Native Assets (`litertlm_dart 0.19.1+1`):** todas as plataformas nativas (Android, Linux e Windows) executam o **LiteRT-LM `v0.19.1` (`native-v0.19.1-c`, C API 1.0)** diretamente via FFI, sem JRE nem servidor gRPC.
- **Catálogo Aberto e Gestão de Armazenamento:** lista dinamicamente todos os modelos abertos (sem necessidade de token no Hugging Face) de `LiteRtLmModels.all`, exibindo tamanho, capacidades (**Imagem**, **Thinking**), selo **Baixado**, aba de filtro **Todos / Baixados** e botão de exclusão (`FlutterEdgeAi.uninstallModel`) com confirmação para liberar espaço.
- **Instalador Nativo via Catálogo Oficial:** downloads sob demanda gerenciados por `FlutterEdgeAi.installModelFromCatalog(...)` com revisões pinadas no Hugging Face, validação automática de integridade de tamanho (`ModelValidationException`), cancelamento via `CancelToken`, **Foreground Service com notificação persistente de progresso no Android** (`POST_NOTIFICATIONS`, `FOREGROUND_SERVICE_DATA_SYNC`, `SystemForegroundService` e botões/canal em português) e armazenamento seguro fora de `Documents`/OneDrive no Windows (`%LOCALAPPDATA%\flutter_gemma`) e Linux (`~/.local/share/<app>/flutter_gemma`).
- **Sessões Independentes (`openChat`) e Sampling Nativo:** utiliza `InferenceModel.openChat(...)` com rewind in-place ao parar geração (preservando imagens), defaults de sampling embarcados no `.litertlm` (Gemma 4: `temperature: 1.0, topK: 64, topP: 0.95`) e tratamento dedicado de `ContextLimitException` e `LiteRtLmBundleException`.
- **Backends por Plataforma + Fallback Transparente:**
  - **Linux (`x86_64` / `arm64`):** `CPU` e `GPU` (Vulkan; FP32 automático acima de 2.048 tokens no Gemma 4).
  - **Windows (`x86_64`):** `CPU`, `GPU` (DirectX 12 / WebGPU com DXC embutido) e `NPU` (Intel Lunar Lake / Panther Lake via OpenVINO `2026.4.1`, embarcado por padrão).
  - **Android (`arm64-v8a`, `minSdk 30`):** `CPU` e `GPU` (OpenCL) embarcados por padrão (mantendo o APK enxuto); NPUs (Qualcomm Snapdragon / Google Tensor) são opt-in via `hooks: user_defines: litertlm_dart:` no `pubspec.yaml` quando utilizadas variantes compiladas por SoC.
  - Em bundles genéricos ou quando o acelerador solicitado não está disponível, o runtime lê o cabeçalho do `.litertlm` e realiza fallback automático (`NPU → GPU → CPU`), exibindo o `activeBackend` real na barra superior.
- **Janela de Contexto (`maxTokens`) e Thinking Mode Configuráveis:** o usuário escolhe entre presets de `1K` a `32K` tokens (limites oficiais dos bundles Gemma 4 E2B/E4B) e ativa/desativa o **Modo Pensamento**, com bloco retrátil de raciocínio em cada resposta.
- **Geração em Segundo Plano no Android (`InferenceForegroundService`):** durante a inferência, um Foreground Service (`shortService` no Android 14+) com `PARTIAL_WAKE_LOCK` mantém o pensamento e a geração ativos mesmo ao minimizar o app ou bloquear a tela, exibindo notificação silenciosa (**"Pensando na resposta..."** / **"Gerando resposta..."**) com botão **"Parar"** e notificação **"Resposta pronta"** caso o modelo termine em segundo plano.

### 4. Requisitos por Plataforma

- **SDK:** Flutter `>= 3.44.0` e Dart `>= 3.12.0`.
- **Android:** `minSdk 30`, arquitetura `arm64-v8a` (emuladores `x86_64` e dispositivos 32-bit não suportam `.litertlm`).
- **Linux (`x86_64` / `arm64`):** dependências de build `clang cmake ninja-build libgtk-3-dev lld`; para GPU, `libvulkan1` + driver Vulkan do fabricante (NVIDIA/AMD/Intel; `llvmpipe` faz fallback para CPU).
- **Windows (`x86_64`):** Visual Studio com workload *Desktop development with C++* e `tar.exe` (nativo no Windows 10/11). Usuários finais não precisam instalar JRE nem VC++ Redistributable adicional.

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

3. **Executar testes unitários e de widget:**
   ```bash
   flutter test
   ```

4. **Executar a aplicação:**
   ```bash
   flutter run -d linux   # ou -d windows / -d android
   ```
