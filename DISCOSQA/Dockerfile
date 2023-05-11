FROM python:3.8-buster

# Install OpenJDK-8
RUN apt-get update
RUN apt-get install -y openjdk-11-jdk && \
    apt-get clean;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

# Downloading and installing Maven
# 1- Define a constant with the version of maven you want to install
ARG MAVEN_VERSION=3.8.6

# 2- Define a constant with the working directory
ARG USER_HOME_DIR="/root"

# 3- Define the SHA key to validate the maven download
ARG SHA=b4880fb7a3d81edd190a029440cdf17f308621af68475a4fe976296e71ff4a4b546dd6d8a58aaafba334d309cc11e638c52808a4b0e818fc0fd544226d952544

# 4- Define the URL where maven can be downloaded from
ARG BASE_URL=https://apache.osuosl.org/maven/maven-3/${MAVEN_VERSION}/binaries

# 5- Create the directories, download maven, validate the download, install it, remove downloaded file and set links
RUN mkdir -p /usr/share/maven /usr/share/maven/ref \
  && echo "Downloading maven" \
  && curl -fsSL -o /tmp/apache-maven.tar.gz ${BASE_URL}/apache-maven-${MAVEN_VERSION}-bin.tar.gz \
  \
  #&& echo "Checking download hash" \
  #&& echo "${SHA}  /tmp/apache-maven.tar.gz" | sha512sum -c - \
  \
  && echo "Unziping maven" \
  && tar -xzf /tmp/apache-maven.tar.gz -C /usr/share/maven --strip-components=1 \
  \
  && echo "Cleaning and setting links" \
  && rm -f /tmp/apache-maven.tar.gz \
  && ln -s /usr/share/maven/bin/mvn /usr/bin/mvn

#

#COPY ./app .
COPY ./app/requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# 6- Define environmental variables required by Maven, like Maven_Home directory and where the maven repo is located
ENV MAVEN_HOME /usr/share/maven
ENV MAVEN_CONFIG "$USER_HOME_DIR/.m2"
#RUN echo $ python -c "print('Real Python')"
#RUN echo $ python -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")'
#RUN mvn dependency:copy-dependencies -DoutputDirectory=./jars -f ($ python3 -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")')
#RUN mvn dependency:copy-dependencies -DoutputDirectory=./jars -f python -c "import sutime as _;import pathlib; print(pathlib.Path(_.__path__[0]+ '/pom.xml'))"
RUN mvn dependency:copy-dependencies -DoutputDirectory=./jars -f /usr/local/lib/python3.8/site-packages/sutime/pom.xml

WORKDIR app/

COPY ./app .

#EXPOSE 8501
CMD ["streamlit", "run", "app.py"]