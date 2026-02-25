FROM polusai/bfio:2.5.0

ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

WORKDIR ${EXEC_DIR}

# Copy the tool directly
COPY . ${EXEC_DIR}/apply-flatfield-tool

# Install directly from the copied path
RUN pip3 install "${EXEC_DIR}/apply-flatfield-tool" --no-cache-dir

ENTRYPOINT ["python3", "-m", "polus.images.transforms.images.apply_flatfield"]
CMD ["--help"]