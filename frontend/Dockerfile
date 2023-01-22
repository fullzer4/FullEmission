FROM node:16-alpine

WORKDIR /website

COPY package.json yarn.lock ./

RUN yarn install --frozen-lockfile

COPY . .

RUN yarn build

EXPOSE 5173:5173

CMD [ "yarn", "start" ]