# LaTeX Development Container
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install TeX Live full distribution and additional tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-full \
    latexmk \
    biber \
    chktex \
    texlive-lang-cyrillic \
    texlive-fonts-extra \
    texlive-science \
    cm-super \
    dvipng \
    ghostscript \
    imagemagick \
    poppler-utils \
    python3-pygments \
    wget \
    curl \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Quarto (for presentations)
ARG QUARTO_VERSION="1.4.553"
RUN wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb \
    && dpkg -i quarto-${QUARTO_VERSION}-linux-amd64.deb \
    && rm quarto-${QUARTO_VERSION}-linux-amd64.deb

# Set up locale for proper Unicode support
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Reset frontend
ENV DEBIAN_FRONTEND=dialog

