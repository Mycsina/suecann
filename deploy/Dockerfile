# Stage 1: Build WASM + frontend
FROM rust:1.88-alpine AS builder

RUN apk add --no-cache \
    bash \
    curl \
    unzip \
    binaryen \
    nodejs \
    npm \
    musl-dev

# Install wasm-pack and wasm-bindgen-cli (via cargo for correct platform binary)
RUN cargo install wasm-bindgen-cli --version 0.2.122 && \
    npm install -g wasm-pack && \
    wasm-pack --version

# Install bun
RUN curl -fsSL https://bun.sh/install | bash

# Add wasm32 target
RUN rustup target add wasm32-unknown-unknown

WORKDIR /build

# Copy workspace manifests first (layer caching)
COPY Cargo.toml Cargo.lock ./
COPY src/sueca_solver/Cargo.toml src/sueca_solver/
COPY src/sueca_wann/Cargo.toml src/sueca_wann/
COPY src/sueca_wasm/Cargo.toml src/sueca_wasm/

# Dummy source files to cache dependency compilation
RUN mkdir -p src/sueca_solver/src src/sueca_wann/src src/sueca_wasm/src && \
    echo 'fn main() {}' > src/sueca_solver/src/lib.rs && \
    echo 'fn main() {}' > src/sueca_wann/src/lib.rs && \
    echo 'fn main() {}' > src/sueca_wasm/src/lib.rs && \
    cd src/sueca_wasm && RUSTFLAGS="" wasm-pack build --target web --out-dir /tmp/wasm-dummy --release 2>/dev/null || true

# Copy actual source
COPY src/ src/

# Build WASM
RUN cd src/sueca_wasm && RUSTFLAGS="" wasm-pack build --target web --out-dir ../../frontend/src/wasm --release

# Build frontend
COPY frontend/ frontend/
RUN cd frontend && ~/.bun/bin/bun install && ~/.bun/bin/bun x vite build


# Stage 2: Serve static files
FROM nginx:alpine

COPY --from=builder /build/frontend/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
