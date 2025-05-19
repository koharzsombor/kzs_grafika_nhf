//=============================================================================================
// Kohár Zsombor Q8EPW6 - Számítógépes Grafika Nagyházi feladat 
// 
// Inspitrációk:
// https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models
// https://www.youtube.com/watch?v=ja8yCvXzw2c
// https://www.youtube.com/watch?v=PH9q0HNBjT4
// 
// Fizikához felhasznált forrás:
// Ian Millington - Game Physics Engine Development
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
		gl_Position = MVP * vec4(vtxPos, 1); // to NDC
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
		vec3 texColor = vec3(1, 1, 1);//texture(diffuseTexture, texcoord).rgb;
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
const float ro = 1000.0f; //Víz
const float Epsilon = 0.00001f;

const mat4 IDENTITY = mat4(
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1
);

const mat3 IDENTITY3D = mat3(
	1, 0, 0,
	0, 1, 0,
	0, 0, 1
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

struct Quaternion {
	float r, i, j, k;

	Quaternion(float real, float imaginary, float jimaginary, float kimaginary) : r(real), i(imaginary), j(jimaginary), k(kimaginary) {}

	Quaternion operator*(const Quaternion& q) const {
		return Quaternion(
			r * q.r - i * q.i - j * q.j - k * q.k,
			r * q.i + i * q.r + j * q.k - k * q.j,
			r * q.j - i * q.k + j * q.r + k * q.i,
			r * q.k + i * q.j - j * q.i + k * q.r
		);
	}

	Quaternion operator+(const Quaternion& q) const {
		return Quaternion(r + q.r, i + q.i, j + q.j, k + q.k);
	}

	Quaternion operator*(float s) const {
		return Quaternion(r * s, i * s, j * s, k * s);
	}

	void normalize() {
		float m = sqrtf(r * r + i * i + j * j + k * k);

		if (m > Epsilon) {
			r /= m;
			i /= m;
			j /= m;
			k /= m;
		}
	}

	mat4 toRotationMatrix() const {
		return mat4(
			1.0f - 2.0f * (j * j + k * k), 2.0f * (i * j - k * r), 2.0f * (i * k + j * r), 0.0f,
			2.0f * (i * j + k * r), 1.0f - 2.0f * (i * i + k * k), 2.0f * (j * k - i * r), 0.0f,
			2.0f * (i * k - j * r), 2.0f * (j * k + i * r), 1.0f - 2.0f * (i * i + j * j), 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
	}

	mat3 toRotationMatrix3D() const {
		return mat3(
			1.0f - 2.0f * (j * j + k * k), 2.0f * (i * j - k * r), 2.0f * (i * k + j * r),
			2.0f * (i * j + k * r), 1.0f - 2.0f * (i * i + k * k), 2.0f * (j * k - i * r),
			2.0f * (i * k - j * r), 2.0f * (j * k + i * r), 1.0f - 2.0f * (i * i + j * j)
		);
	}
};

class PhongShader : public GPUProgram {
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

			setUniform(phase, "phase" + index);
			setUniform(dir, "direction" + index);

			phases[i] = phase;
			directions[i] = dir;
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
		Use();
		t += dt / 10.0f;
		setUniform(t, "t");
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

class OBJSurface : public Mesh {
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
	virtual void Draw(RenderState state) {
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

struct PhysicsObject {
	virtual void Simulate(float dt) = 0;
};

class Movable {
protected:
	vec3 position;
	Quaternion rotation;

	mat4 M() const {
		mat4 Translate = translate(position);
		mat4 Rotate = rotation.toRotationMatrix();

		return Translate * Rotate;
	}

	mat4 MInv() const {
		mat4 InverseTranslate = translate(-position);
		mat4 InverseRotate = transpose(rotation.toRotationMatrix());

		return InverseRotate * InverseTranslate;
	}
	 
	void SetModelingTransformation(RenderState& state) {
		state.M = M();
		state.Minv = MInv();
	}
public:
	Movable(const vec3& pos) : position(pos), rotation(1, 0, 0, 0) {}
};

class BuoyVoxel {
	float volume;
	float a;
	vec3 position;
public:
	BuoyVoxel(const vec3& pos, float sideLength) : position(pos), a(sideLength) {
		volume = a * a * a;
	}

	void simulate(const mat4& M, const vec3& wCOM , vec3& force, vec3& torque, bool stationary) const {
		vec4 wP = M * vec4(position.x, position.y, position.z, 1);
		vec3 wPos = vec3(wP.x, wP.y, wP.z);

		WaterShader* waterShader = WaterSurface::shader;

		vec3 waterPos;
		vec3 waterNormal;

		waterShader->getWater(wPos, waterPos, waterNormal);
		float bottom = wPos.y - a / 2.0f;
		float waterDifference = waterPos.y - bottom;
		float waterPercent = waterDifference / a;

		if (waterPercent > 1.0f) {
			waterPercent = 1.0f;
		}
		else if (waterPercent < 0.0f) {
			waterPercent = 0.0f;
		}

		float wtr = waterPercent * volume;

		force = stationary ? vec3(0, 1, 0) * (wtr * ro * g) : waterNormal * ((wtr * ro * g)); //magyar neve: felható erõ
		torque = cross(force, wPos - wCOM); //magyar neve: forgató nyomaték
	}
};

class BuoyCube : public PhysicsObject, public Movable {
	float volume;
	float m;
	float a;
	float numOfVoxels;
	std::vector<BuoyVoxel> voxels;
	mat3 invI; //magyar neve: Inverz tehetetlenségi tenzor
	vec3 v = vec3(0); //magyar neve: sebesség
	vec3 omega = vec3(0); //magyar neve: szögsebesség

public:
	bool stationary = true;

	BuoyCube(const vec3& pos, float sideLength, int voxelCount, float mass) : a(sideLength), numOfVoxels(voxelCount), m(mass), Movable(pos) {
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

		invI = mat3(6.0f / (m * a * a));
	}

	void Collide(BuoyCube& other) {
		float r1 = a * 0.75f;
		float r2 = other.a * 0.75f;

		vec3 dir = position - other.position;
		float distance = length(dir);

		if (distance >= r1 + r2)
			return;

		dir = normalize(dir);
		
		float vAlongDir1 = dot(v, dir);
		float vAlongDir2 = dot(other.v, dir);
		
		v += (vAlongDir2 - vAlongDir1) * dir;
		other.v += (vAlongDir1 - vAlongDir2) * dir;
	}

	void Simulate(float dt) override {
		vec3 bouyantForce(0);
		vec3 torque(0);

		mat4 Model = M();

		for (const BuoyVoxel& bouyVoxel : voxels) {
			vec3 buoyForce(0);
			vec3 buoyTorque(0);

			bouyVoxel.simulate(Model, position, buoyForce, buoyTorque, stationary);

			bouyantForce += buoyForce;
			torque += buoyTorque;
		}
		vec3 totalForce = bouyantForce - vec3(0, m * g, 0);

		vec3 acc = totalForce / m;
		v += acc * dt;
		v *= 0.985f;
		position += v * dt;

		mat3 wInvI = rotation.toRotationMatrix3D() * invI * transpose(rotation.toRotationMatrix3D());
		vec3 alpha = wInvI * torque;
		omega += alpha * dt; 
		omega *= 0.98f;
		Quaternion q = Quaternion(0, omega.x, omega.y, omega.z);
		rotation = rotation + (q * rotation) * 0.5f * dt;
		rotation.normalize();
	}
};

class Teapot : public BuoyCube, public OBJSurface {
	Material* material;
public:
	static PhongShader* shader;

	Teapot(const vec3& pos) : BuoyCube(pos, 2.0f, 8, 4500), OBJSurface("Teapot.obj", 1.0f) {
		material = new Material(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), 200.0f);
	}

	void Draw(RenderState state) override {
		SetModelingTransformation(state);
		state.material = material;
		shader->Bind(state);
		OBJSurface::Draw(state);
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
	std::vector<Teapot*> teapots;
	Light light;
	Camera* camera;
public:
	void Build() {
		camera = new Camera;

		vec4 lightDirection(5, 2, 5, 1);
		vec3 Le(2, 2, 2), La(0.4f, 0.4f, 0.4f);
		light = Light(La, Le, lightDirection);

		vec3 eye = vec3(40, 40, 40), vup = vec3(0, 1, 0), lookat = vec3(20, 0, 20);
		float fov = 45 * (float)M_PI / 180;
		camera->set(eye, lookat, vup, fov);

		Material* waterMock = new Material(vec3(0.0f, 0.05f, 0.1f), vec3(0.8f, 0.9f, 1.0f), 200.0f);
		surface = new WaterSurface(40, 40, vec3(), waterMock);

		teapots.push_back(new Teapot(vec3(20, 2, 20)));
		teapots.push_back(new Teapot(vec3(15, 2, 15)));
		teapots.push_back(new Teapot(vec3(25, 2, 25)));
		teapots.push_back(new Teapot(vec3(10, 2, 30)));
		teapots.push_back(new Teapot(vec3(20, 2, 30)));
	}

	void Render() {
		RenderState state;
		state.wEye = camera->eye;
		state.V = camera->V();
		state.P = camera->P();
		state.light = light;
		surface->Draw(state);
		for (Teapot* teapot : teapots) {
			teapot->Draw(state);
		}
	}

	void Animate(float dt) {
		surface->Animate(dt);
		for (Teapot* teapot : teapots) {
			teapot->Simulate(dt);
		}

		for (size_t i = 0; i < teapots.size(); i++) {
			for (size_t j = i + 1; j < teapots.size(); j++) {
				teapots[i]->Collide(*teapots[j]);
			}
		}
	}

	void RoateCamera(float dt) {
		camera->Animate(dt);
	}

	void UnfreezeObjects() {
		for (Teapot* teapot : teapots) {
			teapot->stationary = false;
		}
	}
};

WaterShader* WaterSurface::shader;
PhongShader* Teapot::shader;

class Waterloo : public glApp {
	Scene* scene;
public:
	Waterloo() : glApp("Waterloo") {}

	// Inicializáció, 
	void onInitialization() {
		std::srand(time(NULL));
		WaterSurface::shader = new WaterShader;
		Teapot::shader = new PhongShader;

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
		case ' ':
			scene->UnfreezeObjects();
			break;
		default:
			break;
		}
	}

	void onTimeElapsed(float startTime, float endTime) {
		float dt = endTime - startTime;
		const float minDelta = 0.01f;

		for (float t = startTime; t < endTime; t += minDelta) {
			float timeDelta = fmin(minDelta, dt);
			scene->Animate(timeDelta);
		}
		refreshScreen();
	}
};

Waterloo app;

