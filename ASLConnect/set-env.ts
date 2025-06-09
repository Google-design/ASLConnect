const fs = require('fs');
const dotenv = require('dotenv');
const env = dotenv.config().parsed;

const targetPath = './src/environments/environment.ts';
const targetPathProd = './src/environments/environment.prod.ts';

const envFileContent = `
export const environment = {
  production: false,
  firebaseConfig: {
    apiKey: '${env.NG_APP_API_KEY}',
    authDomain: '${env.NG_APP_AUTH_DOMAIN}',
    projectId: '${env.NG_APP_PROJECT_ID}',
    storageBucket: '${env.NG_APP_STORAGE_BUCKET}',
    messagingSenderId: '${env.NG_APP_MESSAGING_SENDER_ID}',
    appId: '${env.NG_APP_APP_ID}',
    measurementId: '${env.NG_APP_MEASUREMENT_ID}'
  }
};`;

fs.writeFileSync(targetPath, envFileContent);
fs.writeFileSync(targetPathProd, envFileContent);
