import { ExternalLink, Code, Terminal, Server } from "lucide-react";

// Update this URL with your actual GitHub repository URL
const GITHUB_REPO_URL = "https://github.com/yourusername/your-repo";

export function BackendSetupGuide() {

  return (
    <div className="p-6 bg-gradient-to-br from-[#ff3358]/20 via-[#00ffc6]/20 to-[#ff8038]/20 border border-[#00ffc6]/30 rounded-lg">
      <div className="flex items-start gap-4 mb-4">
        <div className="p-3 bg-[#00ffc6]/20 rounded-lg">
          <Server className="w-6 h-6 text-[#00ffc6]" />
        </div>
        <div className="flex-1">
          <h3 className="text-xl font-semibold text-white mb-2">
            Backend API Not Available
          </h3>
          <p className="text-white/80 mb-4">
            The frontend is unable to connect to the backend API. Go to the GitHub link below to see how to start the backend server.
          </p>
        </div>
      </div>

      <div className="space-y-4">
        {/* GitHub Link - Prominent */}
        <div className="p-6 bg-gradient-to-r from-[#00ffc6]/20 to-[#00e6b8]/20 border-2 border-[#00ffc6]/50 rounded-lg">
          <div className="flex items-center gap-3 mb-3">
            <Code className="w-6 h-6 text-[#00ffc6]" />
            <h4 className="text-lg font-semibold text-white">Setup Instructions</h4>
          </div>
          <p className="text-base text-white/90 mb-4 font-medium">
            Go to this GitHub link to see how to start the backend server:
          </p>
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-3 px-6 py-3 bg-[#00ffc6] text-[#0f1117] rounded-lg font-bold text-lg hover:bg-[#00e6b8] transition-all transform hover:scale-105 shadow-lg"
          >
            <ExternalLink className="w-5 h-5" />
            <span>Open GitHub Repository</span>
          </a>
          <p className="text-sm text-white/70 mt-3">
            The repository contains complete setup instructions, documentation, and all the code you need to get the backend running.
          </p>
        </div>

        {/* Quick Reference Instructions */}
        <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
          <div className="flex items-center gap-3 mb-3">
            <Terminal className="w-5 h-5 text-[#ff3358]" />
            <h4 className="font-semibold text-white">Quick Reference (See GitHub for Full Instructions)</h4>
          </div>
          <p className="text-sm text-white/70 mb-3">
            For detailed setup instructions, troubleshooting, and more, visit the GitHub repository linked above.
          </p>
          
          <div className="space-y-3 text-sm text-white/80">
            <div>
              <p className="font-semibold text-white mb-1">1. Install Python Dependencies</p>
              <code className="block p-2 bg-[#0f1117] border border-white/10 rounded text-[#00ffc6] mt-1">
                pip install -r requirements.txt
              </code>
            </div>

            <div>
              <p className="font-semibold text-white mb-1">2. Generate Dashboard Cache (Optional but Recommended)</p>
              <code className="block p-2 bg-[#0f1117] border border-white/10 rounded text-[#00ffc6] mt-1">
                python scripts/build_dashboard_cache.py
              </code>
              <p className="text-xs text-white/60 mt-1">
                This pre-generates track DNA, championship state, and coach data for faster loading.
              </p>
            </div>

            <div>
              <p className="font-semibold text-white mb-1">3. Start the Backend API Server</p>
              <code className="block p-2 bg-[#0f1117] border border-white/10 rounded text-[#00ffc6] mt-1">
                python -m uvicorn src.api.dashboard_api:app --reload --port 8000
              </code>
              <p className="text-xs text-white/60 mt-1">
                Keep this terminal open. The API will be available at <span className="text-[#00ffc6]">http://localhost:8000</span>
              </p>
            </div>

            <div>
              <p className="font-semibold text-white mb-1">4. Verify Backend is Running</p>
              <p className="text-xs text-white/60 mt-1">
                Open <a href="http://localhost:8000/api/status" target="_blank" rel="noopener noreferrer" className="text-[#00ffc6] hover:underline">http://localhost:8000/api/status</a> in your browser. 
                You should see a status response.
              </p>
            </div>

            <div>
              <p className="font-semibold text-white mb-1">5. Refresh This Page</p>
              <p className="text-xs text-white/60 mt-1">
                Once the backend is running, refresh this page to connect to the API.
              </p>
            </div>
          </div>
        </div>

        {/* Additional Notes */}
        <div className="p-4 bg-[#ff3358]/10 border border-[#ff3358]/30 rounded-lg">
          <h4 className="font-semibold text-white mb-2">⚠️ Troubleshooting</h4>
          <ul className="text-sm text-white/80 space-y-1 list-disc list-inside">
            <li>Make sure Python 3.9+ is installed</li>
            <li>Ensure port 8000 is not in use by another application</li>
            <li>Check that all dependencies are installed correctly</li>
            <li>Verify the API server is running in a separate terminal</li>
            <li>Check browser console for detailed error messages</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

