# My Hero 🦸‍♂️

Aplicativo Flutter multiplataforma (**Android**, **Linux** e **Windows**) para interação e execução local de modelos de Inteligência Artificial **Google Gemma 4** (E2B e E4B) utilizando `flutter_gemma` (`1.11.4`) + `flutter_gemma_litertlm` (`1.8.6`) via **Dart FFI + Native Assets**, com suporte a streaming, **Thinking Mode**, seleção de **janela de contexto (`maxTokens`)**, escolha de backend (**CPU / GPU / NPU**) e envio de imagens.

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
            │   │   └── gemma_local_datasource.dart            # Comunicação com flutter_gemma + LiteRtLmEngine (download nativo e FFI)
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
                    ├── model_selector_widget.dart             # Seleção CPU/GPU/NPU, maxTokens (1K–32K), Thinking e download nativo
                    └── model_switch_dialog.dart               # Modal para troca dinâmica de modelo/backend/contexto/thinking
```

---

## 🛠️ Tecnologias e Padrões Adotados

### 1. Injeção de Dependências (`get_it`)
Localizada em `lib/src/core/di/injection.dart`.
- **Singletons:** Serviços e fontes de dados com ciclo de vida único (`MediaPickerService`, `GemmaLocalDatasource`, `ChatHistoryDatasource`, `ChatRepository`).
- **Factories:** Instâncias criadas sob demanda por tela/ciclo de vida (`ChatController`).

### 2. Gerenciamento de Estado Reativo (`signals_flutter`)
Os estados das features utilizam **Signals** para reatividade granular (`SignalBuilder` e `effect`).

### 3. Engine Modular LiteRT-LM (`flutter_gemma 1.11.4` + `flutter_gemma_litertlm 1.8.6`)
- **Dart FFI + Native Assets:** todas as plataformas nativas (Android, Linux e Windows) executam o LiteRT-LM (`v0.17.1-a`) diretamente via C API, sem JRE nem servidor gRPC.
- **Instalador Nativo de Modelos:** downloads gerenciados por `FlutterGemma.installModel(...).fromNetwork(...)` com verificação idempotente, cancelamento via `CancelToken`, modo foreground no Android e armazenamento seguro fora de `Documents`/OneDrive no Windows (`%LOCALAPPDATA%\flutter_gemma`) e Linux (`~/.local/share/<app>/flutter_gemma`).
- **Backends por Plataforma + Fallback Transparente:**
  - **Linux:** `CPU` e `GPU` (Vulkan).
  - **Windows (`x86_64`):** `CPU`, `GPU` (DirectX 12 / WebGPU com DXC embutido) e `NPU` (Intel LunarLake/PantherLake).
  - **Android (`arm64-v8a`, `minSdk 30`):** `CPU`, `GPU` (OpenCL) e `NPU` (Qualcomm Snapdragon).
  - Caso o acelerador solicitado não esteja disponível, o runtime realiza fallback automático (`NPU → GPU → CPU`) e o app exibe o `activeBackend` real na barra superior.
- **Janela de Contexto (`maxTokens`) e Thinking Mode Configuráveis:** o usuário escolhe entre presets de `1K` a `32K` tokens (limites oficiais dos bundles Gemma 4 E2B/E4B) e ativa/desativa o **Modo Pensamento**, com bloco retrátil de raciocínio em cada resposta.

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
