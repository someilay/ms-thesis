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

# Install PSCyr fonts (Times with T2A/Cyrillic support)
ARG PSCYR_URL="https://github.com/AndreyAkinshin/Russian-Phd-LaTeX-Dissertation-Template/blob/master/PSCyr/pscyr0.4d.zip?raw=true"
RUN curl -fsSL -o /tmp/pscyr.zip "$PSCYR_URL" \
    && unzip -o /tmp/pscyr.zip -d /tmp/pscyr \
    && mkdir -p /usr/local/share/texmf/fonts \
    && cp -r /tmp/pscyr/PSCyr/fonts/* /usr/local/share/texmf/fonts/ \
    && mkdir -p /usr/local/share/texmf/tex/latex/pscyr \
    && cp /tmp/pscyr/PSCyr/tex/latex/pscyr/* /usr/local/share/texmf/tex/latex/pscyr/ \
    && mktexlsr \
    && updmap-sys --enable Map=pscyr.map \
    && rm -rf /tmp/pscyr /tmp/pscyr.zip

# Reset frontend
ENV DEBIAN_FRONTEND=dialog

