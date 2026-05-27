FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/xmem-go ./cmd/xmem

FROM alpine:3.21
WORKDIR /app
COPY --from=build /out/xmem-go /app/xmem-go
COPY .env.example /app/.env.example
EXPOSE 8081
CMD ["/app/xmem-go"]

