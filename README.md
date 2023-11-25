Reconhecimento Facial e de Mãos com TensorFlow no React
Visão Geral
Este projeto é uma aplicação ReactJS desenvolvida para estudos e insights em reconhecimento facial e de mãos usando TensorFlow. A aplicação utiliza o TensorFlow.js juntamente com modelos específicos para facemesh e handpose. Ao acessar sua webcam, ela mapeia pontos-chave em seu rosto e mãos em tempo real.

Como Começar
Para executar o projeto localmente, siga estas etapas simples:

Clone o repositório em sua máquina local:

bash
Copy code
git clone https://github.com/darlanquimas/face-landmarks.git
Navegue até o diretório do projeto:

bash
Copy code
cd face-landmarks
Instale as dependências do projeto usando Yarn ou npm:

bash
Copy code
yarn install

# ou

npm install
Inicie o servidor de desenvolvimento:

bash
Copy code
yarn start

# ou

npm start
Abra seu navegador da web e acesse a aplicação em http://localhost:3000.

Dependências
O projeto depende das seguintes bibliotecas:

@tensorflow-models/facemesh: Versão 0.0.5
@tensorflow-models/handpose: Versão 0.1.0
@tensorflow/tfjs: Versão 4.13.0
@testing-library/jest-dom: Versão 5.14.1
@testing-library/react: Versão 13.0.0
@testing-library/user-event: Versão 13.2.1
react: Versão 18.2.0
react-dom: Versão 18.2.0
react-scripts: Versão 5.0.1
react-webcam: Versão 7.2.0
web-vitals: Versão 2.1.0
Uso
Ao executar a aplicação, abra sua webcam e experimente o reconhecimento facial e de mãos em tempo real. O aplicativo é o resultado de experimentação e aprendizado sobre as capacidades do TensorFlow.js para reconhecer características faciais e movimentos das mãos.

Observações
Este projeto está em andamento, e alguns toques adicionais ainda estão pendentes.
O objetivo principal é compreender a funcionalidade do TensorFlow no reconhecimento facial e de mãos.
Sinta-se à vontade para explorar, experimentar e contribuir para o projeto. Feliz codificação! 🚀
