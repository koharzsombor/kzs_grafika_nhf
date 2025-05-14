//=============================================================================================
// Kohár Zsombor Q8EPW6 - Számítógépes Grafika Nagyházi feladat 
// 
// Inspitrációk:
// https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models
// https://www.youtube.com/watch?v=ja8yCvXzw2c
// https://www.youtube.com/watch?v=PH9q0HNBjT4
//=============================================================================================
#include "framework.h"


const char* phongVertexSource = R"(
	#version 330

	struct Light {
		vec3 La, Le;
		vec4 wLightPos;
	};

	uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
	uniform Light light;    // light sources 
	uniform vec3  wEye;         // pos of eye
	layout(location = 0) in vec3  vtxPos;            // pos in modeling space
	layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
	layout(location = 2) in vec2  vtxUV;
	out vec3 wNormal;		    // normal in world space
	out vec3 wView;             // view in world space
	out vec3 wLight;		    // light dir in world space
	out vec2 texcoord;
		
	void main() {
		gl_Position = MVP * vec4(vtxPos), 1); // to NDC
		vec4 wPos = M * vec4(vtxPos, 1);
		wLight = light.wLightPos.xyz * wPos.w - wPos.xyz * light.wLightPos.w;
		wView  = wEye.xyz * wPos.w - wPos.xyz;
		wNormal = (vec4(vtxNorm, 0) * Minv).xyz;
		texcoord = vtxUV;
	}
)";

const char* phongFragmentSource = R"(
	#version 330

	struct Light {
		vec3 La, Le;
		vec4 wLightPos;
	};

	struct Material {
		vec3 kd, ks, ka;
		float shininess, emission;
	};

	uniform Material material;
	uniform Light light;    // light sources 
	uniform sampler2D diffuseTexture;
	in  vec3 wNormal;       // interpolated world sp normal
	in  vec3 wView;         // interpolated world sp view
	in  vec3 wLight;        // interpolated world sp illum dir
	in  vec2 texcoord;
    out vec4 outColor;      // output goes to frame buffer

	void main() {
		vec3 N = normalize(wNormal);
		vec3 V = normalize(wView); 
		vec3 texColor = vec3(1, 1, 1);
		vec3 ka = material.ka * texColor;
		vec3 kd = material.kd * texColor;
		vec3 emission = material.emission * texColor;

		vec3 L = normalize(wLight.xyz);
		vec3 H = normalize(L + V);
		float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
		vec3 radiance = emission + ka * light.La + (kd * cost + material.ks * pow(cosd, material.shininess)) * light.Le;
		outColor = vec4(radiance, 1);
	}
)";

const char* waterVertexSource = R"(
	#version 330

	struct Light {
		vec3 La, Le;
		vec4 wLightPos;
	};

	uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
	uniform Light light;    // light sources 
	uniform vec3  wEye;         // pos of eye
	layout(location = 0) in vec3  vtxPos;            // pos in modeling space
	layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
	layout(location = 2) in vec2  vtxUV;
	out vec3 wNormal;		    // normal in world space
	out vec3 wView;             // view in world space
	out vec3 wLight;		    // light dir in world space
	out vec2 texcoord;

	uniform float t;

	const int numOfWaves = 32;

	const float startingFrequency = 0.9;
	const float startingAmplitude = 0.7;	

	const float amplitudeDelta = 0.72; // Minden esetben 0 < x < 1
	const float frequencyDelta = 1.18; // Minden esetben 1 < x

	uniform float phase[numOfWaves];
	uniform vec2 direction[numOfWaves];

	void main() {
		float height = 0.0;

		float currentFrequency = startingFrequency;
		float currentAplitude = startingAmplitude;
		float fxSum = 0.0;

		float dxSum = 0.0;
		float dzSum = 0.0;

		for (int i = 0; i < numOfWaves; ++i) {
			float wavePhase = dot(vtxPos.xz, direction[i]) * currentFrequency + t * phase[i];
			
			float fx = currentAplitude * exp(sin(wavePhase) - 1);
			float dx = currentFrequency * direction[i].x * fx * cos(wavePhase);
			float dz = currentFrequency * fx * cos(wavePhase) * direction[i].y;
			
			fxSum += fx;
			dxSum += dx;
			dzSum += dz;

			currentFrequency *= frequencyDelta;
			currentAplitude *= amplitudeDelta;
		}

		vec3 P = vec3(vtxPos.x, fxSum, vtxPos.z);
		vec3 N = normalize(vec3(-dxSum, 1.0, -dzSum));

		gl_Position = MVP * vec4(P, 1); // to NDC
		vec4 wPos = M * vec4(vtxPos, 1);
		wLight = light.wLightPos.xyz * wPos.w - wPos.xyz * light.wLightPos.w;
		wView  = wEye.xyz * wPos.w - wPos.xyz;
		wNormal = (vec4(N, 0) * Minv).xyz;
		texcoord = vtxUV;
	}
)";

const char* waterFragmentSource = R"(
	#version 330

	struct Light {
		vec3 La, Le;
		vec4 wLightPos;
	};

	struct Material {
		vec3 kd, ks, ka;
		float shininess, emission;
	};

	uniform Material material;
	uniform Light light;    // light sources 
	uniform sampler2D diffuseTexture;
	in  vec3 wNormal;       // interpolated world sp normal
	in  vec3 wView;         // interpolated world sp view
	in  vec3 wLight;        // interpolated world sp illum dir
	in  vec2 texcoord;
    out vec4 outColor;      // output goes to frame buffer

	void main() {
		vec3 N = normalize(wNormal);
		vec3 V = normalize(wView); 
		vec3 texColor = vec3(1, 1, 1);//texture(diffuseTexture, texcoord).rgb;
		vec3 ka = material.ka * texColor;
		vec3 kd = material.kd * texColor;
		vec3 emission = material.emission * texColor;

		vec3 L = normalize(wLight.xyz);
		vec3 H = normalize(L + V);
		float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
		vec3 radiance = emission + ka * light.La + (kd * cost + material.ks * pow(cosd, material.shininess)) * light.Le;
		
		vec3 normalColor = N * 0.5 + 0.5;
	
		outColor = vec4(radiance, 1);
	}
)";

const int winWidth = 600, winHeight = 600;
const float g = 9.8f;
const float ro = 1.0f; //Víz

const mat4 IDENTITY = mat4(
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1
);

const int waveCount = 32;

struct Material {
	vec3 kd = vec3(1, 1, 1), ks = vec3(0, 0, 0), ka = vec3(1, 1, 1);
	float shininess = 1, emission = 0;

	Material(vec3 _kd, vec3 _ks, float _shininess)
		: ka(_kd* (float)M_PI), kd(_kd), ks(_ks) {
		shininess = _shininess;
	}
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};

struct RenderState {
	mat4	        MVP, M, Minv, V, P;
	Material* material = nullptr;
	Light			light;
	Texture* texture = nullptr;
	vec3	        wEye;
};

struct VtxData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};


//---------------------------
class PhongShader : public GPUProgram {
	//---------------------------
public:
	PhongShader() : GPUProgram(phongVertexSource, phongFragmentSource) {}

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.P * state.V * state.M, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(state.material->kd, "material.kd");
		setUniform(state.material->ks, "material.ks");
		setUniform(state.material->ka, "material.ka");
		setUniform(state.material->shininess, "material.shininess");
		setUniform(state.material->emission, "material.emission");
		setUniform(state.light.La, "light.La");
		setUniform(state.light.Le, "light.Le");
		setUniform(state.light.wLightPos, "light.wLightPos");
	}
};

/*
	Proceduláris generáció változóinak beálításának a módjához ötletet adott:
	https://github.com/GarrettGunnell/Water/blob/main/Assets/Scripts/Water.cs
*/
class WaterShader : public GPUProgram {
	float t = 0;

	float phases[waveCount];
	vec2 directions[waveCount];
	const float medianWavelength = 1.0f;
	const float wavelengthRange = 1.0f;
	const float medianDirection = 0.0f;
	const float directionalRange = 360.0f * (M_PI / 180.0f);
	const float medianAmplitude = 1.0f;
	const float medianSpeed = 1.0f;
	const float speedRange = 0.1f;

public:
	WaterShader() : GPUProgram(waterVertexSource, waterFragmentSource) { GenerateWaveParams(); }

	void GenerateWaveParams() {
		float wavelengthMin = medianWavelength / (1.0f + wavelengthRange);
		float wavelengthMax = medianWavelength * (1.0f + wavelengthRange);
		float directionMin = medianDirection - directionalRange;
		float directionMax = medianDirection + directionalRange;
		float speedMin = fmax(0.01f, medianSpeed - speedRange);
		float speedMax = medianSpeed + speedRange;
		float ampOverLen = medianAmplitude / medianWavelength;

		for (int i = 0; i < waveCount; i++) {
			float waveLength = wavelengthMin + static_cast<float>(std::rand()) / RAND_MAX * (wavelengthMax - wavelengthMin);
			float direction = directionMin + static_cast<float>(std::rand()) / RAND_MAX * (directionMax - directionMin);
			float speed = speedMin + static_cast<float>(std::rand()) / RAND_MAX * (speedMax - speedMin);

			float amplitude = waveLength * ampOverLen;
			float frequency = 2.0f / waveLength;
			vec2 dir = normalize(vec2(cosf(direction), sinf(direction)));
			float phase = speed * sqrtf(g * 2.0f * M_PI / waveLength);

			std::string index = "[" + std::to_string(i) + "]";

			printf("%f, %f\n", dir.x, dir.y);
			setUniform(phase, "phase" + index);
			setUniform(dir, "direction" + index);
		}
	}

	//cpu simulation
	void getWater(const vec3& wP, vec3& P, vec3& N) const {
		const float startingFrequency = 0.9;
		const float startingAmplitude = 0.7;

		const float amplitudeDelta = 0.72; // Minden esetben 0 < x < 1
		const float frequencyDelta = 1.18; // Minden esetben 1 < x

		float height = 0.0;

		float currentFrequency = startingFrequency;
		float currentAplitude = startingAmplitude;
		float fxSum = 0.0;

		float dxSum = 0.0;
		float dzSum = 0.0;

		for (int i = 0; i < waveCount; ++i) {
			float wavePhase = dot(vec2(wP.x, wP.z), directions[i]) * currentFrequency + t * phases[i];

			float fx = currentAplitude * exp(sin(wavePhase) - 1);
			float dx = currentFrequency * directions[i].x * fx * cos(wavePhase);
			float dz = currentFrequency * fx * cos(wavePhase) * directions[i].y;

			fxSum += fx;
			dxSum += dx;
			dzSum += dz;

			currentFrequency *= frequencyDelta;
			currentAplitude *= amplitudeDelta;
		}

		P = vec3(wP.x, fxSum, wP.z);
		N = normalize(vec3(-dxSum, 1.0, -dzSum));
	}

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.P * state.V * state.M, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(state.material->kd, "material.kd");
		setUniform(state.material->ks, "material.ks");
		setUniform(state.material->ka, "material.ka");
		setUniform(state.material->shininess, "material.shininess");
		setUniform(state.material->emission, "material.emission");
		setUniform(state.light.La, "light.La");
		setUniform(state.light.Le, "light.Le");
		setUniform(state.light.wLightPos, "light.wLightPos");
	}

	void SetTime(float dt) {
		t += dt;
		setUniform(t / 10, "t");
	}
};

//---------------------------
class Mesh : public Geometry<VtxData> {
	//---------------------------
public:
	virtual void Draw(RenderState state) = 0;
	Mesh() {
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VtxData), (void*)offsetof(VtxData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VtxData), (void*)offsetof(VtxData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VtxData), (void*)offsetof(VtxData, texcoord));
	}
};

//---------------------------
class OBJSurface : public Mesh {
	//---------------------------
public:
	OBJSurface(std::string pathname, float scale) {
		std::vector<vec3> vertices, normals;
		std::vector<vec2> uvs;
		std::ifstream read;
		char line[256];
		read.open(pathname);
		if (!read.is_open()) {
			printf("%s cannot be opened\n", pathname.c_str());
		}
		while (!read.eof()) {
			read.getline(line, 256);
			float x, y, z;
			if (sscanf(line, "v %f %f %f\n", &x, &y, &z) == 3) {
				vertices.push_back(vec3(x * scale, y * scale, z * scale));
				continue;
			}
			if (sscanf(line, "vn %f %f %f\n", &x, &y, &z) == 3) {
				normals.push_back(vec3(x, y, z));
				continue;
			}
			if (sscanf(line, "vt %f %f\n", &x, &y) == 2) {
				uvs.push_back(vec2(x, y));
				continue;
			}
			int v[4], t[4], n[4];
			VtxData vd[4];
			if (sscanf(line, "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n",
				&v[0], &t[0], &n[0], &v[1], &t[1], &n[1], &v[2], &t[2], &n[2], &v[3], &t[3], &n[3]) == 12) {
				for (int i = 0; i < 4; ++i) {
					vd[i].position = vertices[v[i] - 1]; vd[i].texcoord = uvs[t[i] - 1]; vd[i].normal = normals[n[i] - 1];
				}
				vtx.push_back(vd[0]); vtx.push_back(vd[1]); vtx.push_back(vd[2]);
				vtx.push_back(vd[0]); vtx.push_back(vd[2]); vtx.push_back(vd[3]);
			}
			if (sscanf(line, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
				&v[0], &t[0], &n[0], &v[1], &t[1], &n[1], &v[2], &t[2], &n[2]) == 9) {
				for (int i = 0; i < 3; ++i) {
					vd[i].position = vertices[v[i] - 1]; vd[i].texcoord = uvs[t[i] - 1]; vd[i].normal = normals[n[i] - 1];
					vtx.push_back(vd[i]);
				}
			}
		}
		read.close();
		updateGPU();
	}
	void Draw(RenderState state) {
		Bind();
		glDrawArrays(GL_TRIANGLES, 0, vtx.size());
	}
};

//---------------------------
class ParamSurface : public Mesh {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }
	virtual VtxData GenVertexData(float u, float v) = 0;

	void create(int N = 500, int M = 500) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtx.push_back(GenVertexData((float)j / M, (float)i / N));
				vtx.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		updateGPU();
	}
	virtual void Draw(RenderState state) override {
		Bind();
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class WaterSurface : public ParamSurface {
	float width, height, time;
public:
	//vec3 position;
	Material* material;
	static WaterShader* shader;
	WaterSurface(float w, float h, const vec3& pos, Material* m) : width(w), height(h), material(m) {
		create();
	}

	VtxData GenVertexData(float u, float v) override {
		VtxData vd = VtxData();

		vd.normal = vec3(0, 1, 0);
		vd.position = u * width * vec3(1, 0, 0) + v * height * vec3(0, 0, 1);
		vd.texcoord = vec2(u, v);

		return vd;
	}

	void SetModelingTransformation(RenderState& state) {
		state.M = IDENTITY; //translate(position);
		state.Minv = IDENTITY;//translate(-position);
	}

	void Draw(RenderState state) override {
		SetModelingTransformation(state);
		state.material = material;
		shader->Bind(state);
		ParamSurface::Draw(state);
	}

	void Animate(float dt) {
		shader->SetTime(dt);
	}
};

class BuoyVoxel {
	float volume;
	float a;
	float m;
	vec3 position, centerOfMass;
	int voxelCount;
public:
	BuoyVoxel(const vec3& pos, float sideLength) : position(pos), a(sideLength) {
		volume = a * a * a;
	}

	void simulate(const mat4& M, vec3& force, vec3& torque) {
		vec3 wPos;

		WaterShader* waterShader = WaterSurface::shader;

		vec3 waterPos;
		vec3 waterNormal;

		waterShader->getWater(wPos, waterPos, waterNormal);
		float bottom = wPos.y - a / 2.0f;
		float waterDifference = bottom - waterPos.y;
		float waterPercent = waterDifference / a;

		if (waterPercent > 1.0f) {
			waterPercent = 1.0f;
		}
		else if (waterPercent < 0.0f) {
			waterPercent = 0.0f;
		}

		float wtr = waterPercent * volume;

		force = waterNormal * ((wtr * ro * g) - (m * g));
		torque = cross(force, position - centerOfMass);
	}
};

class BuoyCube {
	float volume;
	float m;
	float a;
	float numOfVoxels;
	vec3 position;
	std::vector<BuoyVoxel> voxels;
	mat4 I;

	float V() {
		return 0.0f;
	}

public:
	BuoyCube(const vec3& pos, float sideLength, int voxelCount, float mass) : position(pos), a(sideLength), numOfVoxels(voxelCount), m(mass) {
		vec3 bottomCorner = vec3(-1, -1, -1) * (sideLength / 2.0f);
		float voxelSize = sideLength / (float)voxelCount;
		vec3 voxelStart = bottomCorner + vec3(voxelSize / 2.0f, voxelSize / 2.0f, voxelSize / 2.0f);

		for (int i = 0; i < voxelCount; i++)
		{
			for (int j = 0; j < voxelCount; j++)
			{
				for (int k = 0; k < voxelCount; k++)
				{
					vec3 voxelPos = voxelStart + vec3(i * voxelSize, j * voxelSize, k * voxelSize);
					voxels.push_back(BuoyVoxel(voxelPos, voxelSize));
				}
			}
		}
	}

	float v() {
		return ro * V() * g - (m * g);
	}
};

class Camera {
	vec3 lookat, right, up;
	float fov, asp;
public:
	vec3 eye;
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat; fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		asp = (float)winWidth / (float)winHeight;
		right = normalize(cross(vup, w)) * (float)windowSize * asp;
		up = normalize(cross(w, right)) * windowSize;
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cosf(dt) + d.z * sinf(dt), d.y, -d.x * sinf(dt) + d.z * cosf(dt)) + lookat;
		set(eye, lookat, vec3(0, 1, 0), fov);
	}

	mat4 V() const { return lookAt(eye, lookat, up); };
	mat4 P() const { return perspective(fov, asp, 0.1f, 200.0f); };
};

class Scene {
	int current = 0;
	WaterSurface* surface;
	Light light;
	Camera* camera;
public:
	void Build() {
		camera = new Camera;

		vec4 lightDirection(5, 2, 5, 1);
		vec3 Le(2, 2, 2), La(0.4f, 0.4f, 0.4f);
		light = Light(La, Le, lightDirection);

		vec3 eye = vec3(15, 15, 15), vup = vec3(0, 1, 0), lookat = vec3(5, 0, 5);
		float fov = 45 * (float)M_PI / 180;
		camera->set(eye, lookat, vup, fov);

		Material* waterMock = new Material(vec3(0.0f, 0.05f, 0.1f), vec3(0.8f, 0.9f, 1.0f), 200.0f);
		surface = new WaterSurface(10, 10, vec3(-2.5f, -0.5f, -2.5f), waterMock);
	}

	void Render() {
		RenderState state;
		state.wEye = camera->eye;
		state.V = camera->V();
		state.P = camera->P();
		state.light = light;
		surface->Draw(state);
	}

	void Animate(float dt) {
		surface->Animate(dt);
	}

	void RoateCamera(float dt) {
		camera->Animate(dt);
	}
};

WaterShader* WaterSurface::shader;

class Waterloo : public glApp {
	Scene* scene;
public:
	Waterloo() : glApp("Waterloo") {}

	// Inicializáció, 
	void onInitialization() {
		std::srand(time(NULL));
		WaterSurface::shader = new WaterShader;

		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		scene = new Scene;
		scene->Build();
	}

	// Ablak újrarajzolás
	void onDisplay() {
		glClearColor(0.5f, 0.5f, 0.5f, 0);     // háttér szín
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, winWidth, winHeight);
		scene->Render();
	}

	void onKeyboard(int key) {
		switch (key)
		{
		case 'a':
			scene->RoateCamera(-M_PI / 40.0f);
			refreshScreen();
			break;
		case 'd':
			scene->RoateCamera(M_PI / 40.0f);
			refreshScreen();
			break;
		case 's':
			scene->Animate(M_PI / 4.0f);
			refreshScreen();
		default:
			break;
		}
	}

	void onTimeElapsed(float startTime, float endTime) {
		float dt = startTime - endTime;
		scene->Animate(dt);
		//scene->RoateCamera(dt / 10);
		refreshScreen();
	}
};

Waterloo app;

