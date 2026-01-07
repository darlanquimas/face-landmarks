# Reconhecimento Facial e de Mãos com TensorFlow no React + TypeScript

## Visão Geral

Este projeto é uma aplicação React com TypeScript desenvolvida para estudos e insights em reconhecimento facial e de mãos usando TensorFlow.js. A aplicação utiliza modelos modernos do TensorFlow.js para detectar e mapear pontos-chave faciais e das mãos em tempo real através da webcam.

## Tecnologias

- **React 19** - Biblioteca JavaScript para construção de interfaces
- **TypeScript 5.3** - Superset do JavaScript com tipagem estática
- **TensorFlow.js 4.15** - Machine Learning no navegador
- **@tensorflow-models/face-landmarks-detection** - Detecção de landmarks faciais
- **@tensorflow-models/hand-pose-detection** - Detecção de pose das mãos
- **MediaPipe** - Framework de ML para detecção em tempo real

## Funcionalidades

- ✅ Detecção de landmarks faciais em tempo real (468 pontos)
- ✅ Detecção de pose das mãos (21 pontos por mão)
- ✅ Visualização de malha triangular facial
- ✅ Interface responsiva e moderna
- ✅ Indicadores de carregamento
- ✅ Tratamento de erros robusto
- ✅ Performance otimizada com requestAnimationFrame

## Como Começar

### Pré-requisitos

- Node.js 18+ e npm ou yarn
- Navegador moderno com suporte a WebGL
- Webcam

### Instalação

Clone o repositório e navegue até o diretório do projeto:

```bash
git clone https://github.com/darlanquimas/face-landmarks.git
cd face-landmarks
```

Instale as dependências:

```bash
npm install --legacy-peer-deps
```

ou

```bash
yarn install
```

> **Nota:** O projeto usa `--legacy-peer-deps` devido a conflitos de dependências entre os modelos TensorFlow. Um arquivo `.npmrc` foi configurado para aplicar isso automaticamente.

### Executar

Inicie o servidor de desenvolvimento:

```bash
npm start
```

ou

```bash
yarn start
```

Abra seu navegador e acesse [http://localhost:3000](http://localhost:3000).

## Dependências Principais

- [@tensorflow-models/face-landmarks-detection](https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection) - Modelo moderno para detecção facial
- [@tensorflow-models/hand-pose-detection](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection) - Modelo para detecção de mãos
- [@tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs) - TensorFlow.js core
- [react-webcam](https://www.npmjs.com/package/react-webcam) - Componente React para acesso à webcam

## Uso

1. Ao abrir a aplicação, permita o acesso à sua webcam quando solicitado
2. Aguarde o carregamento dos modelos de ML (indicado na tela)
3. Posicione-se em frente à câmera
4. A aplicação detectará automaticamente seu rosto e mãos, desenhando os landmarks em tempo real

## Estrutura do Projeto

```
src/
├── App.tsx          # Componente principal com lógica de detecção
├── App.css          # Estilos do componente principal
├── util.ts          # Funções utilitárias para desenho dos landmarks
├── index.tsx        # Ponto de entrada da aplicação
└── index.css        # Estilos globais
```

## Melhorias Implementadas

- ✅ Migração para TypeScript com tipagem forte
- ✅ Atualização para React 19
- ✅ Migração de `facemesh` (deprecado) para `face-landmarks-detection`
- ✅ Migração de `handpose` para `hand-pose-detection`
- ✅ Layout responsivo e moderno
- ✅ Performance otimizada com `requestAnimationFrame`
- ✅ Indicadores visuais de carregamento
- ✅ Tratamento robusto de erros
- ✅ Limpeza adequada de recursos (cleanup)

## Desenvolvimento

Este projeto foi desenvolvido para fins educacionais e de experimentação com TensorFlow.js e detecção de landmarks em tempo real no navegador.
