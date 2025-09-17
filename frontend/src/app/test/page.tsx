export default function TestPage() {
  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Tailwind Test Page</h1>
        
        <div className="bg-white rounded-lg shadow-lg p-6 mb-4">
          <h2 className="text-2xl font-semibold text-gray-800 mb-2">Card Title</h2>
          <p className="text-gray-600 mb-4">If you can see this styled properly, Tailwind is working!</p>
          
          <button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
            Blue Button
          </button>
          
          <button className="ml-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
            Green Button
          </button>
          
          <button className="ml-2 px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600">
            Orange Button
          </button>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-red-500 text-white p-4 rounded">Red Box</div>
          <div className="bg-yellow-500 text-white p-4 rounded">Yellow Box</div>
          <div className="bg-purple-500 text-white p-4 rounded">Purple Box</div>
        </div>
      </div>
    </div>
  );
}