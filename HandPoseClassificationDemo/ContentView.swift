import SwiftUI

struct ContentView: View {
    @StateObject var handPoseClassifier = HandPoseClassifier()
    
    var body: some View {
        Text(handPoseClassifier.predictionResult ?? "Unknown")
            .padding()
        
        if let errorMessage = handPoseClassifier.errorMessage {
            Text(errorMessage)
                .padding()
        }
    }
}
