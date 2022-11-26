pipeline{
    agent any
    stages{
        stage ('Cloning our git'){

          steps{
             git 'https://github.com/Pranjal-Bankawat/ReadMyTaste.git'
          }
        }

        stage ('Building our image'){

          steps{

            script{

                // bat 'docker build -t pranjalbankawat/bookrecommendationsystem .'
                sh 'docker build -t pranjalbankawat/bookrecommendationsystem .'

                // docerImage = dock÷er.build registry + ":$BUILD_NUMBER"

            }

          }  



        }

        stage('Deploy our image') {

            steps {

                script {

                    sh 'docker login -u pranjalbankawat -p bankawatP@22'

                    sh 'docker image push pranjalbankawat/bookrecommendationsystem'
                    // bat 'docker login -u pranjalbankawat -p bankawatP@22'

                    // bat 'docker image push pranjalbankawat/bookrecommendationsystem'

                    

                }

            }

        }

    }

}