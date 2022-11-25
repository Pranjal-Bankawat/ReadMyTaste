// pipeline{
//     agent any
//     stages{
//         stage ('Cloning our git'){
//           steps{
//              git 'https://github.com/Pranjal-Bankawat/ReadMyTaste.git'
//           }
//         }
//         stage ('Building our image'){
//           steps{
//             script{
//                 sh 'sudo docker build -t bookrecommendationsystem .'
//             }
//           }  

//         }
//         stage('Deploy our image') { 
//             steps { 
//                 script { 
//                     docker.withRegistry( '', registryCredential ) { 
//                         dockerImage.push() 
//                     }
//                 } 
//             }
//         } 
//     }
// }

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

                sh 'sudo docker build -t pranjalbankawat/bookrecommendationsystem .'

                // docerImage = dock÷er.build registry + ":$BUILD_NUMBER"

            }

          }  



        }

        stage('Deploy our image') {

            steps {

                script {

                    sh 'docker login -u pranjalbankawat -p bankawatP@22'

                    sh 'docker image push pranjalbankawat/bookrecommendationsystem111'

                    

                }

            }

        }

    }

}