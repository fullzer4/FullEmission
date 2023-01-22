FROM node:16-alpine

WORKDIR /website

COPY package.json yarn.lock ./

RUN yarn install --frozen-lockfile

COPY . .

RUN yarn build

CMD [ "yarn", "start" ]

EXPOSE 3000:3000