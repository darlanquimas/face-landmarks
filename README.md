# Reconhecimento Facial e de Mãos com TensorFlow no React

## Visão Geral

Este projeto é uma aplicação ReactJS desenvolvida para estudos e insights em reconhecimento facial e de mãos usando TensorFlow. A aplicação utiliza o [TensorFlow.js](https://www.tensorflow.org/?hl=pt-br) juntamente com modelos específicos para facemesh e handpose. Ao acessar sua webcam, ela mapeia pontos-chave em seu rosto e mãos em tempo real.

## Como Começar

Para executar o projeto localmente, siga estas etapas simples:

Clone o repositório em sua máquina local e avegue até o diretório do projeto:

```bash
git clone https://github.com/darlanquimas/face-landmarks.git
cd face-landmarks
```

Instale as dependências do projeto e inicie o servidor de desenvolvimento usando Yarn ou npm:

```bash
 yarn install
 yarn start
```

# ou

```bash
npm install
npm start
```

Abra seu navegador da web e acesse a aplicação em http://localhost:3000.

## Dependências

O projeto depende das seguintes bibliotecas:

[tensorflow-models/facemesh](https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection)<br>
[tensorflow-models/handpose](https://github.com/tensorflow/tfjs-models/blob/master/handpose/README.md)<br>
[tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs)<br>
[react-webcam](https://www.npmjs.com/package/react-webcam)<br>

## Uso

Ao executar a aplicação, abra sua webcam e experimente o reconhecimento facial e de mãos em tempo real. O aplicativo é o resultado de experimentação e aprendizado sobre as capacidades do TensorFlow.js para reconhecer características faciais e movimentos das mãos.
