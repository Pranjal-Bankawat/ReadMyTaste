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
                sh 'docker build -t bookrecommendationsystem .'
            }
          }  

        }
        stage('Deploy our image') { 
            steps { 
                script { 
                    docker.withRegistry( '', registryCredential ) { 
                        dockerImage.push() 
                    }
                } 
            }
        } 
    }
}